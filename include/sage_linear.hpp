// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"

#include <vector>


class SageLinear {
private:
    int num_in_features_;
    int num_out_features_;
    Linear linear_self_;
    Linear linear_neigh_;

    CudaHelper *cuda_helper_;

public:
    struct SageLinearGradients {
        matrix<float> self_grads;
        matrix<float> neigh_grads;
    };
    SageLinear();
    SageLinear(int in_features, int out_features, CudaHelper *helper);
    matrix<float> *get_parameters();
    matrix<float> *get_gradients();
    void set_gradients(matrix<float> *grads);
    void set_parameters(matrix<float> *parameters);
    matrix<float> forward(matrix<float> features, matrix<float> aggr);
    SageLinearGradients backward(matrix<float> in_gradients);
    void update_weights(matrix<float> *gradients);
};

class SageLinearChunked {
private:
    std::vector<SageLinear> sage_linear_layers_;
    CudaHelper *cuda_helper_;
    int num_in_features_;
    int num_out_features_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;
    std::vector<int> input_shape_;

public:
    SageLinearChunked(CudaHelper *helper, int num_in_features, int num_out_features, int chunk_size, int num_nodes);
    matrix<float> forward(matrix<float> features, matrix<float> aggr);
    SageLinear::SageLinearGradients backward(matrix<float> in_gradients);
    matrix<float> *get_parameters();
    void set_parameters(matrix<float> *parameters);
    matrix<float> *get_gradients();
    void update_weights(matrix<float> *gradients);
    std::vector<SageLinear> get_layers();
};

#endif
