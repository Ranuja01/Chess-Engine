from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        Extension(
            "Engine_Cython",
            sources=["Engine_Cython.pyx"],
            include_dirs=["."],  # Make sure to include the directory containing GUI.pxd
        )
    ])
)
