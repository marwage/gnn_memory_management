// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "add.hpp"
#include "cuda_helper.hpp"
#include "layer.hpp"
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

class SageLinear {
protected:
    long num_in_features_;
    long num_out_features_;
    CudaHelper *cuda_helper_;
    Linear linear_self_;
    Linear linear_neigh_;
    Matrix<float> y_;
    SageLinearGradients input_gradients_;

public:
    std::string name_;
    SageLinear();
    SageLinear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    void set(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    Matrix<float> *forward(Matrix<float> *features, Matrix<float> *aggr);
    SageLinearGradients *backward(Matrix<float> *in_gradients);
    std::vector<Matrix<float> *> get_parameters();
    std::vector<Matrix<float> *> get_gradients();
};

class SageLinearChunkedParent {
protected:
    long num_in_features_;
    long num_out_features_;
    CudaHelper *cuda_helper_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> *y_;
    SageLinearGradientsChunked input_gradients_;

public:
    std::string name_;

    virtual void set(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) = 0;
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) = 0;
    virtual SageLinearGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients) = 0;
    virtual std::vector<Matrix<float> *> get_parameters() = 0;
    virtual std::vector<Matrix<float> *> get_gradients() = 0;
};

class SageLinearChunked : public SageLinearChunkedParent {
protected:
    LinearChunked linear_self_;
    LinearChunked linear_neigh_;
    AddChunked add_;

public:
    SageLinearChunked();
    SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes);
    void set(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) override;
    SageLinearGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients) override;
    std::vector<Matrix<float> *> get_parameters() override;
    std::vector<Matrix<float> *> get_gradients() override;
};

class SageLinearPipelined : public SageLinearChunkedParent {
protected:
    LinearPipelined linear_self_;
    LinearPipelined linear_neigh_;
    AddPipelined add_;

public:
    SageLinearPipelined();
    SageLinearPipelined(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes);
    void set(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) override;
    SageLinearGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients) override;
    std::vector<Matrix<float> *> get_parameters() override;
    std::vector<Matrix<float> *> get_gradients() override;
};

#endif
