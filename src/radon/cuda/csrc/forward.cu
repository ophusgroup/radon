#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.h"
#include "texture.h"

template<int channels, bool clip_to_circle, typename T>
__global__ void
radon_forward_kernel(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                     RaysCfg cfg) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * channels;

    if (angle_id < cfg.n_angles && ray_id < cfg.det_count) {
        
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }

        // compute ray
        float v, sx, sy, ex, ey;
        v = cfg.height / 2.0;
        sx = (ray_id - cfg.det_count / 2.0f + 0.5f) * cfg.det_spacing;
        sy = 0.71f * cfg.height;
        ex = sx;
        ey = -sy;
        
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;

        if (clip_to_circle) {
            // clip rays to circle (to reduce memory reads)
            const float radius = cfg.det_count / 2.0f;
            const float a = rdx * rdx + rdy * rdy;
            const float b = rsx * rdx + rsy * rdy;
            const float c = rsx * rsx + rsy * rsy - radius * radius;

            // min_clip to 1 to avoid getting empty rays
            const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
            const float alpha_s = (-b - delta_sqrt) / a;
            const float alpha_e = (-b + delta_sqrt) / a;

            rsx += rdx * alpha_s + v;
            rsy += rdy * alpha_s + v;
            rdx *= (alpha_e - alpha_s);
            rdy *= (alpha_e - alpha_s);
        } else {
            // clip to square (to reduce memory reads)
            const float alpha_x_m = (-v - rsx)/rdx;
            const float alpha_x_p = (v - rsx)/rdx;
            const float alpha_y_m = (-v -rsy)/rdy;
            const float alpha_y_p = (v - rsy)/rdy;
            const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
            const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

            if(alpha_s > alpha_e){
                #pragma unroll
                for (int b = 0; b < channels; b++) {
                    if constexpr (std::is_same_v<T, float>) {
                        output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = 0.0f;
                    } else if constexpr (std::is_same_v<T, unsigned short>) {
                        __half half_result = __float2half(0.0f);
                        output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = 
                            *reinterpret_cast<unsigned short*>(&half_result);
                    } else {
                        output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = __float2half(0.0f);
                    }
                }
                return;
            }

            rsx += rdx*alpha_s + v;
            rsy += rdy*alpha_s + v;
            rdx *= (alpha_e - alpha_s);
            rdy *= (alpha_e - alpha_s);
        }
        
        const uint n_steps = __float2uint_ru(::hypot(rdx, rdy));
        const float vx = rdx / n_steps;
        const float vy = rdy / n_steps;
        const float n = ::hypot(vx, vy);

        for (uint j = 0; j <= n_steps; j++) {
            if (channels == 1) {
                accumulator[0] += tex2DLayered<float>(texture, rsx, rsy, blockIdx.z);
            } else {
                float4 read = tex2DLayered<float4>(texture, rsx, rsy, blockIdx.z);
                accumulator[0] += read.x;
                accumulator[1] += read.y;
                accumulator[2] += read.z;
                accumulator[3] += read.w;
            }
            rsx += vx;
            rsy += vy;
        }

        #pragma unroll
        for (int b = 0; b < channels; b++) {
            const float result = accumulator[b] * n;
            // Handle type conversion properly
            if constexpr (std::is_same_v<T, float>) {
                output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = result;
            } else if constexpr (std::is_same_v<T, unsigned short>) {
                // For unsigned short, convert float to __half first, then to unsigned short
                __half half_result = __float2half(result);
                output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = 
                    *reinterpret_cast<unsigned short*>(&half_result);
            } else {
                // For __half type
                output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = __float2half(result);
            }
        }
    }
}

template<typename T>
void radon_forward_cuda(
        const T *x, const float *angles, T *y,
        TextureCache &tex_cache, const RaysCfg &cfg, const int batch_size, const int device
) {
    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = (batch_size % 4 == 0) ? 4 : 1;

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, cfg.height, cfg.width, channels, precision});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(roundup_div(cfg.det_count, 16), roundup_div(cfg.n_angles, 16), batch_size / channels);

    // Handle different channel and clip_to_circle cases with compile-time constants
    if (channels == 4) {
        if (cfg.clip_to_circle) {
            if(is_float){
                radon_forward_kernel<4, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else{
                radon_forward_kernel<4, true> << < grid_dim, block_dim >> >
                                                        ((T *) y, tex->texture, angles, cfg);
            }
        } else {
            if(is_float){
                radon_forward_kernel<4, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else{
                radon_forward_kernel<4, false> << < grid_dim, block_dim >> >
                                                        ((T *) y, tex->texture, angles, cfg);
            }
        }
    } else {
        if (cfg.clip_to_circle) {
            if(is_float){
                radon_forward_kernel<1, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else{
                radon_forward_kernel<1, true> << < grid_dim, block_dim >> >
                                                        ((T *) y, tex->texture, angles, cfg);
            }
        } else {
            if(is_float){
                radon_forward_kernel<1, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else{
                radon_forward_kernel<1, false> << < grid_dim, block_dim >> >
                                                        ((T *) y, tex->texture, angles, cfg);
            }
        }
    }
}

template void
radon_forward_cuda<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache, const RaysCfg &cfg,
                          const int batch_size, const int device);

template void radon_forward_cuda<unsigned short>(const unsigned short *x, const float *angles, unsigned short *y,
                                                 TextureCache &tex_cache, const RaysCfg &cfg,
                                                 const int batch_size, const int device);