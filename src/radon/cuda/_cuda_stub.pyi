from torch import Tensor

def add_arrays(
    input_a: Tensor, input_b: Tensor, output_c: Tensor
) -> None: ...


def subtract_arrays(
    input_a: Tensor, input_b: Tensor, output_c: Tensor
) -> None: ...

def forward(x: Tensor, angles: Tensor, 
                  tex_cache: TextureCache, rays_cfg: RaysCfg) -> Tensor: ...

def backward(x: Tensor, angles: Tensor, 
                   tex_cache: TextureCache, rays_cfg: RaysCfg) -> Tensor: ...

class RaysCfg:
    width: int
    height: int
    det_count: int
    det_spacing: float
    n_angles: int
    clip_to_circle: bool
    
    def __init__(self, width: int, height: int, det_count: int, 
                 det_spacing: float, n_angles: int, clip_to_circle: bool) -> None: ...

class TextureCache:
    def __init__(self, size: int) -> None: ...
    def free(self) -> None: ...