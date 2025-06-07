#include <cuda_fp16.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <vector>

#include "cfg.h"
#include "log.h"
#include "noise.h"
#include "radon.h"
#include "symbolic.h"
#include "texture.h"
#include "utils.h"

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor
symbolic_forward(const symbolic::SymbolicFunction& f,
                 torch::Tensor angles,
                 const ProjectionCfg& proj)
{
  TORCH_CHECK(!angles.device().is_cuda(), "angles must be on CPU");
  CHECK_CONTIGUOUS(angles);

  const int n_angles = angles.size(0);
  auto y = torch::empty({ n_angles, proj.det_count_u });

  symbolic::forward(
    f, proj, angles.data_ptr<float>(), n_angles, y.data_ptr<float>());

  return y;
}

torch::Tensor
symbolic_discretize(const symbolic::SymbolicFunction& f,
                    const int height,
                    const int width)
{
  auto y = torch::empty({ height, width });

  f.discretize(y.data_ptr<float>(), height, width);

  return y;
}

torch::Tensor
radon_forward(torch::Tensor x,
              torch::Tensor angles,
              TextureCache& tex_cache,
              const VolumeCfg vol_cfg,
              const ProjectionCfg proj_cfg,
              const ExecCfg exec_cfg)
{
  CHECK_INPUT(x);
  CHECK_INPUT(angles);

  auto dtype = x.dtype();

  const int batch_size = x.size(0);
  const int n_angles = angles.size(0);
  const int device = x.device().index();

  // allocate output sinogram tensor
  auto options = torch::TensorOptions().dtype(dtype).device(x.device());

  if (vol_cfg.is_3d) {
    auto y = torch::empty(
      { batch_size, n_angles, proj_cfg.det_count_v, proj_cfg.det_count_u },
      options);

    if (dtype == torch::kFloat16) {
      radon::forward_cuda_3d((__half*)x.data_ptr<at::Half>(),
                             angles.data_ptr<float>(),
                             (__half*)y.data_ptr<at::Half>(),
                             tex_cache,
                             vol_cfg,
                             proj_cfg,
                             exec_cfg,
                             batch_size,
                             device);
    } else {
      radon::forward_cuda_3d(x.data_ptr<float>(),
                             angles.data_ptr<float>(),
                             y.data_ptr<float>(),
                             tex_cache,
                             vol_cfg,
                             proj_cfg,
                             exec_cfg,
                             batch_size,
                             device);
    }
    return y;
  } else {
    auto y =
      torch::empty({ batch_size, n_angles, proj_cfg.det_count_u }, options);

    if (dtype == torch::kFloat16) {
      radon::forward_cuda((__half*)x.data_ptr<at::Half>(),
                          angles.data_ptr<float>(),
                          (__half*)y.data_ptr<at::Half>(),
                          tex_cache,
                          vol_cfg,
                          proj_cfg,
                          exec_cfg,
                          batch_size,
                          device);
    } else {
      radon::forward_cuda(x.data_ptr<float>(),
                          angles.data_ptr<float>(),
                          y.data_ptr<float>(),
                          tex_cache,
                          vol_cfg,
                          proj_cfg,
                          exec_cfg,
                          batch_size,
                          device);
    }
    return y;
  }
}

torch::Tensor
radon_backward(torch::Tensor x,
               torch::Tensor angles,
               TextureCache& tex_cache,
               const VolumeCfg& vol_cfg,
               const ProjectionCfg& proj_cfg,
               const ExecCfg& exec_cfg)
{
  CHECK_INPUT(x);
  CHECK_INPUT(angles);

  auto dtype = x.dtype();

  const int batch_size = x.size(0);
  const int device = x.device().index();

  TORCH_CHECK(angles.size(0) <= 4096, "Can only support up to 4096 angles")

  // create output image tensor
  auto options = torch::TensorOptions().dtype(dtype).device(x.device());

  if (vol_cfg.is_3d) {
    auto y = torch::empty(
      { batch_size, vol_cfg.depth, vol_cfg.height, vol_cfg.width }, options);

    if (dtype == torch::kFloat16) {
      radon::backward_cuda_3d((__half*)x.data_ptr<at::Half>(),
                              angles.data_ptr<float>(),
                              (__half*)y.data_ptr<at::Half>(),
                              tex_cache,
                              vol_cfg,
                              proj_cfg,
                              exec_cfg,
                              batch_size,
                              device);
    } else {
      radon::backward_cuda_3d(x.data_ptr<float>(),
                              angles.data_ptr<float>(),
                              y.data_ptr<float>(),
                              tex_cache,
                              vol_cfg,
                              proj_cfg,
                              exec_cfg,
                              batch_size,
                              device);
    }
    return y;
  } else {
    auto y =
      torch::empty({ batch_size, vol_cfg.height, vol_cfg.width }, options);

    if (dtype == torch::kFloat16) {
      radon::backward_cuda((__half*)x.data_ptr<at::Half>(),
                           angles.data_ptr<float>(),
                           (__half*)y.data_ptr<at::Half>(),
                           tex_cache,
                           vol_cfg,
                           proj_cfg,
                           exec_cfg,
                           batch_size,
                           device);
    } else {
      radon::backward_cuda(x.data_ptr<float>(),
                           angles.data_ptr<float>(),
                           y.data_ptr<float>(),
                           tex_cache,
                           vol_cfg,
                           proj_cfg,
                           exec_cfg,
                           batch_size,
                           device);
    }

    return y;
  }
}

void
radon_add_noise(torch::Tensor x,
                RadonNoiseGenerator& noise_generator,
                const float signal,
                const float density_normalization,
                const bool approximate)
{
  CHECK_INPUT(x);

  const int height = x.size(0) * x.size(1);
  const int width = x.size(2);
  const int device = x.device().index();

  noise_generator.add_noise(x.data_ptr<float>(),
                            signal,
                            density_normalization,
                            approximate,
                            width,
                            height,
                            device);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

  m.def("forward", &radon_forward, "Radon forward projection");
  m.def("backward", &radon_backward, "Radon back projection");

  m.def("add_noise", &radon_add_noise, "Add noise to sinogram");

  m.def("symbolic_forward", &symbolic_forward, "TODO");
  m.def("symbolic_discretize", &symbolic_discretize, "TODO");

  m.def("set_log_level", [](const int level) {
    Log::log_level = static_cast<Log::Level>(level);
  });

  py::class_<TextureCache>(m, "TextureCache")
    .def(py::init<size_t>())
    .def("free", &TextureCache::free);

  py::class_<RadonNoiseGenerator>(m, "RadonNoiseGenerator")
    .def(py::init<const uint>())
    .def("set_seed",
         (void(RadonNoiseGenerator::*)(const uint)) &
           RadonNoiseGenerator::set_seed)
    .def("free", &RadonNoiseGenerator::free);

  py::class_<VolumeCfg>(m, "VolumeCfg")
    .def(
      py::init<int, int, int, float, float, float, float, float, float, bool>())
    .def_readonly("depth", &VolumeCfg::depth)
    .def_readonly("height", &VolumeCfg::height)
    .def_readonly("width", &VolumeCfg::width)
    .def_readonly("dx", &VolumeCfg::dx)
    .def_readonly("dy", &VolumeCfg::dy)
    .def_readonly("dz", &VolumeCfg::dz)
    .def_readonly("is_3d", &VolumeCfg::is_3d);

  py::class_<ProjectionCfg>(m, "ProjectionCfg")
    .def(py::init<int, float>())
    .def(py::init<int, float, int, float, float, float, float, float, int>())
    .def("is_2d", &ProjectionCfg::is_2d)
    .def("copy", &ProjectionCfg::copy)
    .def_readonly("projection_type", &ProjectionCfg::projection_type)
    .def_readwrite("det_count_u", &ProjectionCfg::det_count_u)
    .def_readwrite("det_spacing_u", &ProjectionCfg::det_spacing_u)
    .def_readwrite("det_count_v", &ProjectionCfg::det_count_v)
    .def_readwrite("det_spacing_v", &ProjectionCfg::det_spacing_v)
    .def_readwrite("s_dist", &ProjectionCfg::s_dist)
    .def_readwrite("d_dist", &ProjectionCfg::d_dist)
    .def_readwrite("pitch", &ProjectionCfg::pitch)
    .def_readwrite("initial_z", &ProjectionCfg::initial_z)
    .def_readwrite("n_angles", &ProjectionCfg::n_angles);

  py::class_<ExecCfg>(m, "ExecCfg").def(py::init<int, int, int, int>());

  py::class_<symbolic::SymbolicFunction>(m, "SymbolicFunction")
    .def(py::init<float, float>())
    .def("add_gaussian", &symbolic::SymbolicFunction::add_gaussian)
    .def("add_ellipse", &symbolic::SymbolicFunction::add_ellipse)
    .def("move", &symbolic::SymbolicFunction::move)
    .def("scale", &symbolic::SymbolicFunction::scale);
}
