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
public:
    virtual Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr) = 0;
    virtual SageLinearGradients *backward(Matrix<float> *in_gradients) = 0;
    virtual Matrix<float> **get_parameters() = 0;
    virtual void set_parameters(Matrix<float> **parameters) = 0;
    virtual Matrix<float> **get_gradients() = 0;
    virtual void update_weights(Matrix<float> *gradients) = 0;
};

class SageLinear : public SageLinearParent {
private:
    long num_in_features_;
    long num_out_features_;
    Linear linear_self_;
    Linear linear_neigh_;
    SageLinearGradients input_gradients_;
    CudaHelper *cuda_helper_ = NULL;

public:
    SageLinear();
    SageLinear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    void set(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    Matrix<float> **get_parameters() override;
    Matrix<float> **get_gradients() override;
    void set_gradients(Matrix<float> **grads);
    void set_parameters(Matrix<float> **parameters);
    Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr) override;
    SageLinearGradients *backward(Matrix<float> *in_gradients) override;
    void update_weights(Matrix<float> *gradients) override;
};

class SageLinearChunked : public SageLinearParent {
private:
    std::vector<SageLinear> sage_linear_layers_;
    CudaHelper *cuda_helper_ = NULL;
    long num_in_features_;
    long num_out_features_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> features_chunks_;
    std::vector<Matrix<float>> aggr_chunks_;
    std::vector<Matrix<float>> in_gradients_chunks_;
    Matrix<float> y_;
    Matrix<float> self_gradients_;
    Matrix<float> neighbourhood_gradients_;
    SageLinearGradients input_gradients_;

public:
    SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes);
    Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr) override;
    SageLinearGradients *backward(Matrix<float> *in_gradients) override;
    Matrix<float> **get_parameters() override;
    void set_parameters(Matrix<float> **parameters) override;
    Matrix<float> **get_gradients() override;
    void update_weights(Matrix<float> *gradients) override;
    std::vector<SageLinear> *get_layers();
};

#endif
