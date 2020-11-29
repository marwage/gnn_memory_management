// Copyright 2020 Marcel Wagenländer

#ifndef HELPER_HPP
#define HELPER_HPP

#include "sage_linear.hpp"
#include "tensors.hpp"


void save_params(Matrix<float> **parameters);

void save_grads(SageLinearGradients *gradients, Matrix<float> **weight_gradients);

int run_python(std::string module_name, std::string function_name);

int read_return_value(std::string path);

void write_value(int value, std::string path);

int num_equal_rows(Matrix<float> A, Matrix<float> B);

int compare_mat(Matrix<float> *mat_a, Matrix<float> *mat_b, std::string name);

#endif//HELPER_HPP
