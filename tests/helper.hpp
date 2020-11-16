// Copyright 2020 Marcel Wagenländer

#ifndef HELPER_HPP
#define HELPER_HPP

#include "tensors.hpp"
#include "sage_linear.hpp"


void save_params(matrix<float> *parameters);

void save_grads(SageLinearGradients *gradients, matrix<float> *weight_gradients);

matrix<float> gen_rand_matrix(int num_rows, int num_columns);

matrix<float> gen_non_rand_matrix(int num_rows, int num_columns);

int run_python(std::string module_name, std::string function_name);

int read_return_value(std::string path);

int num_equal_rows(matrix<float> A, matrix<float> B);

#endif//HELPER_HPP
