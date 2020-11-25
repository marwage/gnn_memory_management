// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"

#include <vector>


struct SageLinearGradients {
    matrix<float> *self_grads;
    matrix<float> *neigh_grads;
};

class SageLinearParent {
public:
    virtual matrix<float>* forward(matrix<float> *features, matrix<float> *aggr) = 0;
    virtual SageLinearGradients* backward(matrix<float> *in_gradients) = 0;
    virtual matrix<float>** get_parameters() = 0;
    virtual void set_parameters(matrix<float> **parameters) = 0;
    virtual matrix<float>** get_gradients() = 0;
    virtual void update_weights(matrix<float> *gradients) = 0;
};

class SageLinear : public SageLinearParent {
private:
    long num_in_features_;
    long num_out_features_;
    Linear linear_self_;
    Linear linear_neigh_;
    SageLinearGradients input_gradients_;

    CudaHelper *cuda_helper_;

public:
    SageLinear();
    SageLinear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    matrix<float>** get_parameters() override;
    matrix<float>** get_gradients() override;
    void set_gradients(matrix<float> **grads);
    void set_parameters(matrix<float> **parameters);
    matrix<float>* forward(matrix<float> *features, matrix<float> *aggr) override;
    SageLinearGradients* backward(matrix<float> *in_gradients) override;
    void update_weights(matrix<float> *gradients) override;
};

class SageLinearChunked : public SageLinearParent {
private:
    std::vector<SageLinear> sage_linear_layers_;
    CudaHelper *cuda_helper_;
    long num_in_features_;
    long num_out_features_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<matrix<float>> features_chunks_;
    std::vector<matrix<float>> aggr_chunks_;
    std::vector<matrix<float>> in_gradients_chunks_;
    matrix<float> y_;
    matrix<float> self_gradients_;
    matrix<float> neighbourhood_gradients_;
    SageLinearGradients input_gradients_;

public:
    SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes);
    matrix<float>* forward(matrix<float> *features, matrix<float> *aggr) override;
    SageLinearGradients* backward(matrix<float> *in_gradients) override;
    matrix<float>** get_parameters() override;
    void set_parameters(matrix<float> **parameters) override;
    matrix<float>** get_gradients() override;
    void update_weights(matrix<float> *gradients) override;
    std::vector<SageLinear>* get_layers();
};

#endif
