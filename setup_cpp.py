from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="linear_cpp_impl",
    ext_modules=[CppExtension("linear_cpp_impl", ["linear_cpp.cpp"])],
    cmdclass={"build_ext": BuildExtension}
)