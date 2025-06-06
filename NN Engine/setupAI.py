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

# onnx_path = "/home/ranuja/onnxruntime-linux-x64-1.21.0"
eigen_path = "/usr/include/eigen3/"

# removed "nnue.cpp" from sources
# removed os.path.join(onnx_path, "include"), from include_dirs
# removed os.path.join(onnx_path, "lib") from library_dirs
# remove onnxruntime from libraries

# Define the extension module
extensions = [
    Extension(
        "ChessAI",
        sources=["cpp_bitboard.cpp", "threadpool.cpp", "search_engine.cpp", "ChessAI.pyx"],
        language="c++",
        extra_compile_args=[
            "-Ofast",                      # Safe high optimization
            "-march=native",           # Optimize for host CPU
            "-flto",                   # Link-time optimization
            "-fopenmp",                # Multithreading support
            "-g",
            "-fno-omit-frame-pointer",
                      
            "-fno-rtti",               # Removes RTTI overhead
            "-std=c++20",              # Use modern C++
            
            "-mpopcnt", "-mbmi2",      # Enable CPU bit manipulation instructions
        ],
        extra_link_args=[
            "-flto", "-fopenmp", "-pthread"
        ],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],
        include_dirs=[
            np.get_include(),
            eigen_path,
            "." 
        ],
        libraries=[],
        library_dirs=[],
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
