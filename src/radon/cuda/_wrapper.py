from . import _cuda_add


def add_arrays(a, b, c):
    _cuda_add.add_arrays(a, b, c)

def subtract_arrays(a, b, c):
    _cuda_add.subtract_arrays(a, b, c)