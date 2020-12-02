// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"

#include <vector>


struct SageLinearGradients {
    Matrix<float> *self_grads;
    Matrix<float> *neigh_grads;
};

class SageLinearParent {
protected:
    long num_in_features_;
    long num_out_features_;

    CudaHelper *cuda_helper_ = NULL;

    Linear linear_self_;
    Linear linear_neigh_;

    Matrix<float> y_;
    SageLinearGradients input_gradients_;

public:
    virtual Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr) = 0;
    virtual SageLinearGradients *backward(Matrix<float> *in_gradients) = 0;
    Matrix<float> **get_parameters();
    void set_parameters(Matrix<float> **parameters);
    Matrix<float> **get_gradients();
    void set_gradients(Matrix<float> **grads);
    void update_weights(Matrix<float> *gradients);
};

class SageLinear : public SageLinearParent {
private:


public:
    SageLinear();
    SageLinear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    void set(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr) override;
    SageLinearGradients *backward(Matrix<float> *in_gradients) override;
};

class SageLinearChunked : public SageLinearParent {
private:
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> features_chunks_;
    std::vector<Matrix<float>> aggr_chunks_;
    std::vector<Matrix<float>> in_gradients_chunks_;
    Matrix<float> self_gradients_;
    Matrix<float> neighbourhood_gradients_;

public:
    SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes);
    Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr) override;
    SageLinearGradients *backward(Matrix<float> *in_gradients) override;
};

#endif
