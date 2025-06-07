import typing
import warnings

import numpy as np
import torch
import torch.fft

from . import cuda_backend
from .differentiable_functions import RadonForward, RadonBackprojection
from .filtering import FourierFilters
from .projection import Projection
from .utils import normalize_shape, ShapeNormalizer, expose_projection_attributes
from .volumes import Volume2D, Volume3D

warnings.simplefilter("default")


class ExecCfgGeneratorBase:
    def __init__(self):
        pass

    def __call__(self, vol_cfg, proj_cfg, is_half):
        if proj_cfg.projection_type == 2:
            ch = 4 if is_half else 1
            return cuda_backend.ExecCfg(8, 16, 8, ch)

        return cuda_backend.ExecCfg(16, 16, 1, 4)


class BaseRadon:
    def __init__(
        self,
        angles: typing.Union[torch.Tensor, typing.Tuple[int, int, int]],
        volume: typing.Union[Volume2D, Volume3D],
        projection: Projection,
    ):
        # allows angles to be specified as (start_angle, end_angle, n_angles)
        if isinstance(angles, tuple) and len(angles) == 3:
            start_angle, end_angle, n_angles = angles
            angles = np.linspace(
                start_angle,
                end_angle,
                n_angles,
                endpoint=False,
            )

        # make sure that angles are a PyTorch tensor
        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        self.angles = angles
        self.volume = volume
        self.projection = projection
        self.exec_cfg_generator = ExecCfgGeneratorBase()

        # caches used to avoid reallocation of resources
        self.tex_cache = cuda_backend.TextureCache(8)
        self.fourier_filters = FourierFilters()

    def _move_parameters_to_device(self, device):
        if device != self.angles.device:
            self.angles = self.angles.to(device)

    def _check_input(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        if x.dtype == torch.float16:
            assert (
                x.size(0) % 4 == 0
            ), f"Batch size must be multiple of 4 when using half precision. Got batch size {x.size(0)}"

        return x

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor = None,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""Radon forward projection.

        :param x: PyTorch GPU tensor.
        :param angles: PyTorch GPU tensor indicating the measuring angles, if None the angles given to the constructor are used
        :returns: PyTorch GPU tensor containing sinograms.
        """
        x = self._check_input(x)
        self._move_parameters_to_device(x.device)

        angles = angles if angles is not None else self.angles

        shape_normalizer = ShapeNormalizer(self.volume.num_dimensions())
        x = shape_normalizer.normalize(x)

        self.volume.height = x.size(-2)
        self.volume.width = x.size(-1)
        if self.volume.num_dimensions() == 3:
            self.volume.depth = x.size(-3)

        self.projection.cfg.n_angles = len(angles)

        y = RadonForward.apply(
            x,
            angles,
            self.tex_cache,
            self.volume.to_cfg(),
            self.projection.cfg,
            self.exec_cfg_generator,
            exec_cfg,
        )

        return shape_normalizer.unnormalize(y)

    def backward(
        self,
        sinogram,
        angles: torch.Tensor = None,
        volume: typing.Union[Volume2D, Volume3D] = None,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms.
        :param angles: PyTorch GPU tensor indicating the measuring angles, if None the angles given to the constructor
            are used
        :returns: PyTorch GPU tensor containing backprojected volume.
        """
        sinogram = self._check_input(sinogram)
        volume = self.volume if volume is None else volume

        assert (
            volume.has_size()
        ), "Must use forward before calling backward or specify a volume"

        self._move_parameters_to_device(sinogram.device)

        angles = angles if angles is not None else self.angles

        shape_normalizer = ShapeNormalizer(self.volume.num_dimensions())
        sinogram = shape_normalizer.normalize(sinogram)

        self.projection.cfg.n_angles = len(angles)

        y = RadonBackprojection.apply(
            sinogram,
            angles,
            self.tex_cache,
            volume.to_cfg(),
            self.projection.cfg,
            self.exec_cfg_generator,
            exec_cfg,
        )

        return shape_normalizer.unnormalize(y)

    @normalize_shape(2)
    def filter_sinogram(
        self,
        sinogram: torch.Tensor,
        filter_name: typing.Literal[
            "ramp", "shepp-logan", "cosine", "hamming", "hann"
        ] = "ramp",
    ):
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        # Pad sinogram to improve accuracy
        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size
        padded_sinogram = torch.nn.functional.pad(sinogram.float(), (0, pad, 0, 0))
        
        sino_fft = torch.fft.rfft(padded_sinogram, norm="ortho")

        # get filter and apply
        f = self.fourier_filters.get(padded_size, filter_name, sinogram.device)
        filtered_sino_fft = sino_fft * f

        # Inverse fft
        filtered_sinogram = torch.fft.irfft(filtered_sino_fft, norm="ortho")
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

        return filtered_sinogram.to(dtype=sinogram.dtype)


class ParallelBeam(BaseRadon):
    r"""
    |
    .. image:: https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/
            master/pictures/parallelbeam.svg?sanitize=true
        :align: center
        :width: 400px
    |

    Class that implements Radon projection for the Parallel Beam geometry.

    :param det_count: *Required*. Number of rays that will be projected.
    :param angles: *Required*. Array containing the list of measuring angles. Can be a Numpy array, a PyTorch tensor or a tuple
        `(start, end, num_angles)` defining a range.
    :param det_spacing: Distance between two contiguous rays. By default is `1.0`.
    :param volume: Specifies the volume position and scale. By default a uniform volume is used.
        To create a non-uniform volume specify an instance of :class:`torch_radon.Volume2D`.

    """

    def __init__(
        self,
        det_count: int,
        angles: typing.Union[list, np.array, torch.Tensor, tuple],
        det_spacing: float = 1.0,
        volume: Volume2D = None,
    ):
        if volume is None:
            volume = Volume2D()

        projection = Projection.parallel_beam(det_count, det_spacing)

        super().__init__(angles, volume, projection)


expose_projection_attributes(
    ParallelBeam, [("det_count", "det_count_u"), ("det_spacing", "det_spacing_u")]
)



