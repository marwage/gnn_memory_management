// 2020 Marcel Wagenländer

#ifndef ADD_H
#define ADD_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

struct AddGradientsChunked {
    std::vector<Matrix<float>> *a;
    std::vector<Matrix<float>> *b;
};

class Add {
protected:
    std::string name_;
    CudaHelper *cuda_helper_;
    Matrix<float> y_;
    std::vector<Matrix<float> *> gradients_;

public:
    Add();
    Add(CudaHelper *cuda_helper, long num_nodes, long num_features);
    void set(CudaHelper *cuda_helper, long num_nodes, long num_features);
    Matrix<float> *forward(Matrix<float> *a, Matrix<float> *b);
    std::vector<Matrix<float> *> *backward(Matrix<float> *incoming_gradients);
};

class AddChunked {
protected:
    long num_chunks_;
    long chunk_size_;
    long last_chunk_size_;
    CudaHelper *cuda_helper_;
    std::vector<Matrix<float>> y_;
    AddGradientsChunked gradients_;

    bool keep_allocation_;
    float *d_a_;
    float *d_b_;

    void allocate_gpu_memory();
    void free_gpu_memory();

public:
    std::string name_;

    AddChunked();
    AddChunked(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features);
    ~AddChunked();
    virtual void set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features);
    virtual void set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation);
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b);
    virtual AddGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients);
};

class AddPipelined : public AddChunked, public LayerPipelined {
protected:
    long num_steps_;
    std::vector<float *> d_a_;
    std::vector<float *> d_b_;
    std::vector<float *> d_c_;
    std::vector<Matrix<float>> *a_;
    std::vector<Matrix<float>> *b_;

public:
    AddPipelined();
    AddPipelined(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *cudaHelper, long chunkSize, long numNodes, long numFeatures) override;
    void forward_in(long chunk, long buffer) override;
    void forward_out(long chunk, long buffer) override;
    void forward_compute(long chunk, long buffer) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b) override;
    void backward_in(long chunk, long buffer) override;
    void backward_out(long chunk, long buffer) override;
    void backward_compute(long chunk, long buffer) override;
    AddGradientsChunked *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

#endif//ADD_H
