// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_CHUNK_H
#define ALZHEIMER_CHUNK_H

#include "tensors.hpp"

#include <vector>


void chunk_up(Matrix<float> *x, std::vector<Matrix<float>> *x_chunked, long chunk_size);

void stitch(std::vector<Matrix<float>> *x_chunked, Matrix<float> *x);

#endif//ALZHEIMER_CHUNK_H
