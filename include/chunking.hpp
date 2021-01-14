// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_CHUNK_H
#define ALZHEIMER_CHUNK_H

#include "tensors.hpp"

#include <vector>


void init_set_random_values(std::vector<Matrix<float>> *mat, long num_nodes, long num_features, long chunk_size, bool is_row_major);

void chunk_up(Matrix<float> *x, std::vector<Matrix<float>> *x_chunked, long chunk_size);

void stitch(std::vector<Matrix<float>> *x_chunked, Matrix<float> *x);

void double_chunk_up_sp(SparseMatrix<float> *sp_mat, std::vector<SparseMatrix<float>> *chunks, long chunk_size);

#endif//ALZHEIMER_CHUNK_H
