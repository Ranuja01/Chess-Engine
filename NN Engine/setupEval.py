from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module
extensions = [
    Extension(
        "eval_func",                     # Name of the compiled extension
        sources=["cpp_bitboard.cpp", "eval_func.pyx"],       # Source Cython file
        language="c++",                # Use C++ compiler
        extra_compile_args=[
            "-Ofast", "-march=native", "-ffast-math", "-fopenmp",
            "-funroll-loops", "-flto", "-fomit-frame-pointer", "-std=c++20",
            "-fno-math-errno", "-fno-trapping-math", "-fassociative-math",
            "-fno-signed-zeros", "-fno-rounding-math", "-ffp-contract=fast"
        ], # Optimization flags
        extra_link_args=["-flto=16", "-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], 
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
            'infer_types': True,
            'language_level': 3,
        }
    ),
)