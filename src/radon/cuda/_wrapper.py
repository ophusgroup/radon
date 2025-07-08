from . import _cuda_add

def add_arrays(a, b, c):
    _cuda_add.add_arrays(a, b, c)
    return


def subtract_arrays(a, b, c):
    _cuda_add.subtract_arrays(a, b, c)
    return

def forward(x, angles, tex_cache, rays_cfg):
    """Radon forward projection"""
    return _cuda_add.forward(x, angles, tex_cache, rays_cfg)

def backward(x, angles, tex_cache, rays_cfg):
    """Radon backward projection"""
    return _cuda_add.backward(x, angles, tex_cache, rays_cfg)


RaysCfg = _cuda_add.RaysCfg
TextureCache = _cuda_add.TextureCache