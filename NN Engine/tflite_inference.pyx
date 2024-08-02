# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:42:42 2024

@author: Kumodth
"""
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport fopen, fclose, fread, fseek, ftell, SEEK_END, SEEK_SET
from libc.stdio cimport FILE
import numpy as np
cimport numpy as cnp

# TensorFlow Lite C API includes
cdef extern from "tensorflow/lite/c/c_api.h":
    ctypedef struct TfLiteModel
    ctypedef struct TfLiteInterpreter
    ctypedef struct TfLiteInterpreterOptions
    ctypedef struct TfLiteTensor

    TfLiteModel* TfLiteModelCreate(const void* model_data, size_t model_size)
    void TfLiteModelDelete(TfLiteModel* model)

    TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate()
    void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options)
    void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options, int num_threads)

    TfLiteInterpreter* TfLiteInterpreterCreate(TfLiteModel* model, TfLiteInterpreterOptions* optional_options)
    void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter)
    int TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter)
    int TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter)

    TfLiteTensor* TfLiteInterpreterGetInputTensor(TfLiteInterpreter* interpreter, int input_index)
    TfLiteTensor* TfLiteInterpreterGetOutputTensor(TfLiteInterpreter* interpreter, int output_index)

    int TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor, const void* input_data, size_t input_data_size)
    int TfLiteTensorCopyToBuffer(TfLiteTensor* tensor, void* output_data, size_t output_data_size)

    const char* TfLiteVersion()

# Example class to use TensorFlow Lite C API
cdef class TFLiteModel:
    cdef TfLiteModel* model
    cdef TfLiteInterpreter* interpreter
    cdef TfLiteInterpreterOptions* options

    def __cinit__(self):
        self.model = NULL
        self.interpreter = NULL
        self.options = TfLiteInterpreterOptionsCreate()
        if self.options is not NULL:
            TfLiteInterpreterOptionsSetNumThreads(self.options, 2)  # Set number of threads

    def __dealloc__(self):
        if self.interpreter is not NULL:
            TfLiteInterpreterDelete(self.interpreter)
        if self.model is not NULL:
            TfLiteModelDelete(self.model)
        if self.options is not NULL:
            TfLiteInterpreterOptionsDelete(self.options)

    def load_model(self, const char* model_path):
        cdef FILE* f = fopen(model_path, "rb")
        if not f:
            raise IOError("Failed to open model file")

        fseek(f, 0, SEEK_END)
        cdef size_t file_size = ftell(f)
        fseek(f, 0, SEEK_SET)

        cdef void* model_data = malloc(file_size)
        if not model_data:
            fclose(f)
            raise MemoryError("Failed to allocate memory for model buffer")

        fread(model_data, 1, file_size, f)
        fclose(f)

        self.model = TfLiteModelCreate(model_data, file_size)
        if self.model is NULL:
            free(model_data)
            raise RuntimeError("Failed to create TFLite model")

        self.interpreter = TfLiteInterpreterCreate(self.model, self.options)
        if self.interpreter is NULL:
            TfLiteModelDelete(self.model)
            free(model_data)
            raise RuntimeError("Failed to create TFLite interpreter")

        if TfLiteInterpreterAllocateTensors(self.interpreter) != 0:
            raise RuntimeError("Failed to allocate tensors")

    def infer(self, cnp.ndarray[cnp.float32_t, ndim=1] input_data):
        cdef TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(self.interpreter, 0)
        if TfLiteTensorCopyFromBuffer(input_tensor, <const void*>input_data.data, input_data.nbytes) != 0:
            raise RuntimeError("Failed to copy data to input tensor")

        if TfLiteInterpreterInvoke(self.interpreter) != 0:
            raise RuntimeError("Failed to invoke TFLite interpreter")

        cdef TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(self.interpreter, 0)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] output_data = np.zeros((10,), dtype=np.float32)

        if TfLiteTensorCopyToBuffer(output_tensor, <void*>output_data.data, output_data.nbytes) != 0:
            raise RuntimeError("Failed to copy data from output tensor")

        return output_data
