// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"

#include <vector>


struct SageLinearGradients {
    Matrix<float> *self_gradients;
    Matrix<float> *neighbourhood_gradients;
};

struct SageLinearGradientsChunked {
    std::vector<Matrix<float>> *self_gradients;
    std::vector<Matrix<float>> *neighbourhood_gradients;
};

class SageLinearParent {
protected:
    long num_in_features_;
    long num_out_features_;

    CudaHelper *cuda_helper_ = NULL;

    Linear linear_self_;
    Linear linear_neigh_;

public:
    std::vector<Matrix<float> *> get_parameters();
    std::vector<Matrix<float> *> get_gradients();
};

class SageLinear : public SageLinearParent {
private:
    Matrix<float> y_;
    SageLinearGradients input_gradients_;

public:
    SageLinear();
    SageLinear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    void set(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr);
    SageLinearGradients *backward(Matrix<float> *in_gradients);
};

class SageLinearChunked : public SageLinearParent {
private:
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> *features_ = NULL;
    std::vector<Matrix<float>> *aggregated_features_ = NULL;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> self_gradients_;
    std::vector<Matrix<float>> neighbourhood_gradients_;
    SageLinearGradientsChunked input_gradients_;
    Matrix<float> self_weight_sum_;
    Matrix<float> self_bias_sum_;
    Matrix<float> neigh_weight_sum_;
    Matrix<float> neigh_bias_sum_;

public:
    SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr);
    SageLinearGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients);
};

#endif
