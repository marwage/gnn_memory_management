// Copyright 2020 Marcel Wagenländer

#ifndef LINEAR_H
#define LINEAR_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

#include <vector>

class Linear {
protected:
    CudaHelper *cuda_helper_;
    Matrix<float> y_;
    Matrix<float> gradients_;
    long num_nodes_;
    long num_in_features_;
    long num_out_features_;
    Matrix<float> weight_;
    Matrix<float> bias_;
    Matrix<float> bias_expanded_;
    Matrix<float> grad_weight_;
    Matrix<float> grad_bias_;
    std::vector<float> ones_;
    Matrix<float> *x_ = NULL;
    float *d_weight_;
    float *d_ones_;
    float *d_db_;
    float *d_dweight_;

    void init_weight_bias();
    void expand_bias();

public:
    std::string name_;

    Linear();
    Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    void set(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    std::vector<Matrix<float> *> get_parameters();
    std::vector<Matrix<float> *> get_gradients();
    std::vector<float *> get_gradients_cuda();
    void set_gradients(Matrix<float> *weight_grads, Matrix<float> *bias_grads);
    void forward_init();
    void forward_compute(float *d_x, long num_rows, float *d_y);
    void forward_free();
    Matrix<float> *forward(Matrix<float> *x);
    void backward_init();
    void backward_compute(float *d_dy, float *d_x, long num_rows, float *d_dx);
    void backward_free();
    Matrix<float> *backward(Matrix<float> *incoming_gradients);
};

class LinearChunked {
protected:
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    CudaHelper *cuda_helper_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;
    Linear linear_;
    long num_in_features_;
    long num_out_features_;
    std::vector<Matrix<float>> *x_;

public:
    std::string name_;

    LinearChunked();
    LinearChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features);
    virtual void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features);
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x);
    virtual std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients);
    std::vector<Matrix<float> *> get_parameters();
    std::vector<Matrix<float> *> get_gradients();
};

class LinearPipelined : public LinearChunked, public LayerPipelined {
protected:
    long num_steps_;
    std::vector<float *> d_x_;
    std::vector<float *> d_y_;
    std::vector<float *> d_dy_;
    std::vector<float *> d_dx_;
    std::vector<Matrix<float>> *incoming_gradients_;

public:
    LinearPipelined();
    LinearPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features) override;
    void forward_in(long chunk, long buffer) override;
    void forward_out(long chunk, long buffer) override;
    void forward_compute(long chunk, long buffer) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    void backward_in(long chunk, long buffer) override;
    void backward_out(long chunk, long buffer) override;
    void backward_compute(long chunk, long buffer) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

#endif
