from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gsinpaint',
    ext_modules=[
        CUDAExtension('gspaint_cuda', [
            'gspaint_cuda.cpp',
            'gspaint_kernel_bigger_patch.cu'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension}
)