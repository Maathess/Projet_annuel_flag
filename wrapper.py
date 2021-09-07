import numpy as np
import os
from ctypes import *

path_to_dll = "C:/Users/Maathess/Desktop/Projet_annuel_flag/PMC/cmake-build-debug/PMC8.dll"
mylib = cdll.LoadLibrary(path_to_dll)


class MLPWrapper:
    def __init__(self, npl: [int], is_classification: bool = True,
                 alpha: float = 0.01, iterations_count: int = 1000):
        init_size = len(npl)
        init_type = c_int * init_size
        init = init_type(*npl)
        mylib.create_mlp_model.argtypes = [init_type, c_int]
        mylib.create_mlp_model.restype = c_void_p

        self.model = mylib.create_mlp_model(init, int(init_size))
        self.is_classification = is_classification
        self.alpha = alpha
        self.iterations_count = iterations_count

    def fit(self, X, Y):

        if self.is_classification:
            dataset_inputs = np.array(X)
            dataset_expected_outputs = np.array(Y)

            flattened_dataset_inputs = []

            for p in dataset_inputs:
                flattened_dataset_inputs.append(p[0])
                flattened_dataset_inputs.append(p[1])

            # definition de train_classification_stochastic_gradient....
            arrsize_flat = len(flattened_dataset_inputs)
            arrtype_flat = c_float * arrsize_flat
            arr_flat = arrtype_flat(*flattened_dataset_inputs)

            arrsize_exp = len(flattened_dataset_inputs)
            arrtype_exp = c_float * arrsize_exp
            arr_exp = arrtype_exp(*flattened_dataset_inputs)

        if self.is_classification:
            mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.argtypes = [c_void_p, arrtype_flat,
                                                                                                 c_int,
                                                                                                 arrtype_exp, c_float,
                                                                                                 c_int]
            mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.restype = None

            mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(self.model, arr_flat, arrsize_flat,
                                                                                     arr_exp,
                                                                                     self.alpha, self.iterations_count)

            #
            # mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(self.model,
            #                                                                  X.flatten(),
            #                                                                  Y.flatten(),
            #                                                                  self.alpha,
            #                                                                  self.iterations_count)
        else:
            mylib.train_regression_stochastic_gradient_backpropagation_mlp_model(self.model,
                                                                                 X.flatten(),
                                                                                 Y.flatten(),
                                                                                 self.alpha,
                                                                                 self.iterations_count)

    def predict(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X)

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        results = []
        for x in X:
            if self.is_classification:
                results.append(mylib.predict_mlp_model_classification(self.model, x.flatten()))
            else:
                results.append(mylib.predict_mlp_model_regression(self.model, x.flatten()))

        return np.array(results)
