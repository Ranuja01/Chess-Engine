from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Update this path to where TensorFlow Lite C API is installed
tensorflow_include_dir = '/home/ranuja/tensorflow'  # or wherever you installed the headers
tensorflow_lib_dir = '/home/ranuja/tensorflow'  # or wherever you installed the libraries

extensions = [
    Extension(
        "tflite_inference",
        sources=["tflite_inference.pyx"],
        libraries=["tensorflowlite_c", "tensorflow-lite"],  # Include possible alternatives
        library_dirs=[tensorflow_lib_dir],
        include_dirs=[np.get_include(), tensorflow_include_dir],
        language="c++",
        extra_compile_args=["-Ofast", "-Wall"],
        extra_link_args=["-Wl,-rpath," + tensorflow_lib_dir],
    )
]

setup(
    name="tflite_inference",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
