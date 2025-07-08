import torch
from .cuda import RaysCfg

__version__ = "0.1.0"

class Radon:
    def __init__(self):
        self.version = __version__
        self.rays_cfg = RaysCfg(1,1,1,1.0,90,True)

    def get_version(self):
        return self.version