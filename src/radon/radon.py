import torch
from . import cuda_backend
from .utils import normalize_shape
from .differentiable_functions import RadonForward, RadonBackprojection

__version__ = "0.1.0"

class Radon:
    def __init__(self, resolution: int, angles, det_count=-1, det_spacing=1.0, clip_to_circle=False):
        self.version = __version__
        if det_count <= 0:
            det_count = resolution

        rays_cfg = cuda_backend.RaysCfg(resolution, resolution, det_count, det_spacing, len(angles), clip_to_circle)
        self.rays_cfg = rays_cfg

        self.rays_cfg = rays_cfg

        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        # change sign to conform to Astra and Scikit
        self.angles = -angles

        # caches used to avoid reallocation of resources
        self.tex_cache = cuda_backend.TextureCache(8)

        self.det_count = det_count
        self.det_spacing = det_spacing

    def _move_parameters_to_device(self, device):
        if device != self.angles.device:
            self.angles = self.angles.to(device)
    
    def _check_input(self, x, square=False):
        if not x.is_contiguous():
            x = x.contiguous()

        if square:
            assert x.size(1) == x.size(2), f"Input images must be square, got shape ({x.size(1)}, {x.size(2)})."

        if x.dtype == torch.float16:
            assert x.size(
                0) % 4 == 0, f"Batch size must be multiple of 4 when using half precision. Got batch size {x.size(0)}"

        return x
    
    @normalize_shape(2)
    def forward(self, x):
        r"""Radon forward projection.

        :param x: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        :returns: PyTorch GPU tensor containing sinograms. Has shape :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        """
        x = self._check_input(x, square=True)
        self._move_parameters_to_device(x.device)

        return RadonForward.apply(x, self.angles, self.tex_cache, self.rays_cfg)
    
    @normalize_shape(2)
    def backprojection(self, sinogram):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms with shape  :math:`(d_1, \dots, d_n, len(angles), det\_count)`.
        :returns: PyTorch GPU tensor with shape :math:`(d_1, \dots, d_n, r, r)` where :math:`r` is the :attr:`resolution`
            given to the constructor of this class.
        """
        sinogram = self._check_input(sinogram)
        self._move_parameters_to_device(sinogram.device)

        return RadonBackprojection.apply(sinogram, self.angles, self.tex_cache, self.rays_cfg)

    def backward(self, sinogram):
        r"""Same as backprojection"""
        return self.backprojection(sinogram)

    def get_version(self):
        return self.version