from .cuda_backend import ProjectionCfg

class Projection:
    PARALLEL = 0
    FANBEAM = 1
    CONE_FLAT = 2

    def __init__(self, cfg: ProjectionCfg):
        self.cfg = cfg

    @staticmethod
    def parallel_beam(det_count, det_spacing=1.0):
        return Projection(ProjectionCfg(det_count, det_spacing))

    def is_2d(self):
        return self.cfg.is_2d()
