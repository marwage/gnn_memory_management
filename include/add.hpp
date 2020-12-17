// 2020 Marcel Wagenländer

#ifndef ADD_H
#define ADD_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"


struct AddGradients {
    Matrix<float> *a;
    Matrix<float> *b;
};

struct AddGradientsChunked {
    std::vector<Matrix<float>> *a;
    std::vector<Matrix<float>> *b;
};

class Add {
private:
    CudaHelper *cuda_helper_;
    Matrix<float> y_;
    AddGradients gradients_;

public:
    Add(CudaHelper *cuda_helper, long num_nodes, long num_features);
    Matrix<float> *forward(Matrix<float> *a, Matrix<float> *b);
    AddGradients *backward(Matrix<float> *incoming_gradients);
};

class AddChunked {
protected:
    long num_chunks_;
    long chunk_size_;
    long last_chunk_size_;
    CudaHelper *cuda_helper_;
    std::vector<Matrix<float>> y_;
    AddGradientsChunked gradients_;

public:
    AddChunked();
    AddChunked(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features);
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b);
    virtual AddGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients);
};

class AddPipelined : public AddChunked, public LayerPipelined {
protected:
    long num_steps_;
    std::vector<float *> d_a_;
    std::vector<float *> d_b_;
    std::vector<Matrix<float>> *a_;
    std::vector<Matrix<float>> *b_;

public:
    AddPipelined(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features);
    void forward_in(long chunk, long buffer) override;
    void forward_out(long chunk, long buffer) override;
    void forward_compute(long chunk, long buffer) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b);
    AddGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients);
};

#endif//ADD_H
