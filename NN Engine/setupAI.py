# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:29:47 2024

@author: Kumodth
"""

# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

onnx_path = "/home/ranuja/onnxruntime-linux-x64-1.21.0"
eigen_path = "/usr/include/eigen3/"

# Define the extension module
extensions = [
    Extension(
        "ChessAI",                     # Name of the compiled extension
        sources=["cpp_bitboard.cpp", "nnue.cpp", "ChessAI.pyx"],       # Source Cython file
        language="c++",                # Use C++ compiler
        extra_compile_args=[
            "-Ofast", "-march=native", "-ffast-math", "-fopenmp",
            "-funroll-loops", "-flto", "-fomit-frame-pointer", "-std=c++20",
            "-fno-math-errno", "-fno-trapping-math", "-fassociative-math",
            "-fno-signed-zeros", "-fno-rounding-math", "-ffp-contract=fast"
        ], # Optimization flags
        extra_link_args=["-flto=16", "-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        library_dirs=[os.path.join(onnx_path, "lib")],
        libraries=["onnxruntime"],
        include_dirs=[np.get_include(), os.path.join(onnx_path, "include"), eigen_path]
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
