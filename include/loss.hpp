// Copyright 2020 Marcel Wagenländer

#ifndef LOSS_H
#define LOSS_H

#include "tensors.hpp"


class NLLLoss {
private:
    double loss_;
    long num_nodes_;
    bool is_row_major_;
    Matrix<int> *labels_;
    Matrix<float> gradients_;

public:
    NLLLoss(long num_nodes, long num_features);
    float forward(Matrix<float> *x, Matrix<int> *labels);
    Matrix<float> *backward();
};

class NLLLossChunking {
private:
    double loss_;
    long num_nodes_;
    long num_chunks_;
    long chunk_size_;
    long last_chunk_size_;
    bool is_row_major_;
    Matrix<int> *labels_;
    std::vector<Matrix<float>> gradients_;

public:
    NLLLossChunking(long num_nodes, long num_features, long chunk_size);
    float forward(std::vector<Matrix<float>> *x, Matrix<int> *labels);
    std::vector<Matrix<float>> *backward();
};

#endif
