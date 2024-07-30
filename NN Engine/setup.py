# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("chess_eval", ["chess_eval.pyx"]),
]

setup(
    ext_modules=cythonize(extensions),
)
