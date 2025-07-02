from ._wrapper import add_arrays, subtract_arrays

__all__ = ["add_arrays", "subtract_arrays"]

print("CUDA extension loaded successfully.")
print(add_arrays)