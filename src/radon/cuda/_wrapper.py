from . import _cuda_add


def add_arrays(a, b):
    return _cuda_add.add_arrays(a, b)

def subtract_arrays(a, b):
    return _cuda_add.subtract_arrays(a, b)