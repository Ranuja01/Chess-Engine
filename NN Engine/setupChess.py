# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 00:02:05 2024

@author: Kumodth
"""

# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module
extensions = [
    Extension(
        "Cython_Chess",                     # Name of the compiled extension
        sources=["cpp_bitboard.cpp", "Cython_Chess.pyx"],       # Source Cython file
        language="c++",                   # Use C++ compiler
        extra_compile_args=["-Ofast", "-march=native", "-ffast-math", 
        "-funroll-loops", "-flto", "-fomit-frame-pointer", "-std=c++20"], # Optimization flags
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], 
        include_dirs=[np.get_include()],
        
    )
    
]

setup(
    ext_modules=cythonize(extensions),
)
