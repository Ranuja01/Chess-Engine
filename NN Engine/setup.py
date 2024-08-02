# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module
extensions = [
    Extension(
        "ChessAI",                     # Name of the compiled extension
        sources=["ChessAI.pyx"],       # Source Cython file
        language="c++",                   # Use C++ compiler
        extra_compile_args=["-Ofast", "-march=native",  "-ffast-math"], # Optimization flags
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], 
        include_dirs=[np.get_include()]
    )
    
]

setup(
    ext_modules=cythonize(extensions),
)
