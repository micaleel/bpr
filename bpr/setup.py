from setuptools import setup
from setuptools.extension import Extension
import os
import numpy
from glob import glob
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_args = dict(
    include_dirs=[numpy.get_include()],
    libraries=["m"],
    extra_compile_args=[
        "-O3",
        "-Xpreprocessor",
        "-ffast-math",
        "-fopenmp",
        "-march=native",
    ],
    extra_link_args=[
        "-fopenmp",
    ],
)
here = os.path.dirname(__file__)
pyx_files = [x.replace(".pyx", "") for x in glob(os.path.join(here, "*.pyx"))]
print(pyx_files)
ext_modules = [
    Extension(f"{fname}", sources=[f"{fname}.pyx"], **ext_args) for fname in pyx_files
]

setup(
    name="bpr",
    ext_modules=cythonize(
        annotate=True,
        module_list=ext_modules,
    ),
)
