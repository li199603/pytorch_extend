from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="linear_cuda_impl",
    ext_modules=[CUDAExtension("linear_cuda_impl", ["linear_cuda.cpp", "linear_cuda_kernel.cu"])],
    cmdclass={"build_ext": BuildExtension}
)