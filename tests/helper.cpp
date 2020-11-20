// Copyright Marcel Wagenländer 2020

#include "helper.hpp"

#include <Python.h>
#include <iostream>
#include <string>


std::string home = std::getenv("HOME");
std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
std::string test_dir_path = dir_path + "/tests";

void save_params(matrix<float> *parameters) {
    std::string path;

    path = test_dir_path + "/self_weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/self_bias.npy";
    save_npy_matrix(parameters[1], path);
    path = test_dir_path + "/neigh_weight.npy";
    save_npy_matrix(parameters[2], path);
    path = test_dir_path + "/neigh_bias.npy";
    save_npy_matrix(parameters[3], path);
}

void save_grads(SageLinearGradients *gradients, matrix<float> *weight_gradients) {
    std::string path;

    path = test_dir_path + "/self_grads.npy";
    save_npy_matrix(gradients->self_grads, path);
    path = test_dir_path + "/neigh_grads.npy";
    save_npy_matrix(gradients->neigh_grads, path);

    path = test_dir_path + "/self_weight_grads.npy";
    save_npy_matrix(weight_gradients[0], path);
    path = test_dir_path + "/self_bias_grads.npy";
    save_npy_matrix(weight_gradients[1], path);
    path = test_dir_path + "/neigh_weight_grads.npy";
    save_npy_matrix(weight_gradients[2], path);
    path = test_dir_path + "/neigh_bias_grads.npy";
    save_npy_matrix(weight_gradients[3], path);
}

matrix<float> gen_matrix(long num_rows, long num_columns, bool random) {
    long max = 5;

    matrix<float> mat = new_float_matrix(num_rows, num_columns, true);

    for (long i = 0; i < mat.rows * mat.columns; ++i) {
        if (random) {
            mat.values[i] = rand();
        } else {
            mat.values[i] = (float) ((i % max) + 1);
        }
    }

    return mat;
}

matrix<float> gen_rand_matrix(long num_rows, long num_columns) {
        return gen_matrix(num_rows, num_columns, true);
}

matrix<float> gen_non_rand_matrix(long num_rows, long num_columns) {
    return gen_matrix(num_rows, num_columns, false);
}

int run_python(std::string module_name, std::string function_name) {
    std::string path_tests = "/home/ubuntu/gpu_memory_reduction/alzheimer/tests";
    int return_value = 0;

    PyObject *pName, *pModule, *pFunc, *pValue;
    Py_Initialize();

    PyRun_SimpleString(("import sys\nsys.path.append(\"" + path_tests + "\")\n").c_str());
    pName = PyUnicode_DecodeFSDefault(module_name.c_str());
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, function_name.c_str());
        if (pFunc && PyCallable_Check(pFunc)) {
            pValue = PyObject_CallObject(pFunc, NULL);
            if (pValue != NULL) {
                return_value = (int) PyLong_AsLong(pValue);
                Py_DECREF(pValue);
            }
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
    }
    if (Py_FinalizeEx() < 0) {
        std::cout << "Error in Py_FinalizeEx" << std::endl;
    }

    std::cout << "Return value " << return_value << std::endl;
    return return_value;
}

int read_return_value(std::string path) {
    matrix<int> return_mat = load_npy_matrix<int>(path);
    return return_mat.values[0];
}

int num_equal_rows(matrix<float> A, matrix<float> B) {
    int num_rows = 0;
    bool equal_row = true;

    for (int i = 0; i < A.rows; ++i) {
        equal_row = true;
        for (int j = 0; j < A.columns; ++j) {
            if (A.values[j * A.rows + i] != B.values[j * A.rows + i]) {
                equal_row = false;
            }
        }
        if (equal_row) {
            num_rows = num_rows + 1;
        }
    }

    return num_rows;
}
