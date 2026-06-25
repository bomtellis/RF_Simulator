import sys

print(sys.executable)

try:
    import numba

    print("numba", numba.__version__)
    from numba import cuda

    print("cuda imported")
    print("cuda.is_available =", cuda.is_available())
    cuda.detect()
except Exception as e:
    print(type(e).__name__, e)
