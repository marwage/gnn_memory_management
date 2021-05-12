// Copyright 2020 Marcel Wagenländer

#ifndef HELPER_HPP
#define HELPER_HPP

#include "sage_linear.hpp"
#include "tensors.hpp"


void save_params(std::vector<Matrix<float> *> parameters);

void save_grads(SageLinearGradients *gradients, std::vector<Matrix<float> *> weight_gradients);

void save_grads(SageLinearGradientsChunked *gradients, std::vector<Matrix<float> *> weight_gradients, long num_nodes);

int read_return_value(std::string path);

void write_value(int value, std::string path);

int num_equal_rows(Matrix<float> A, Matrix<float> B);

int compare_mat(Matrix<float> *mat_a, Matrix<float> *mat_b, std::string name);

#endif//HELPER_HPP
