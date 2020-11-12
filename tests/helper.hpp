// Copyright 2020 Marcel Wagenländer

#ifndef HELPER_HPP
#define HELPER_HPP

void save_params(matrix<float> *parameters);

void save_grads(SageLinear::SageLinearGradients *gradients, matrix<float> *weight_gradients);

matrix<float> gen_rand_matrix(int num_rows, int num_columns);

int run_python(std::string module_name, std::string function_name);

#endif//HELPER_HPP
