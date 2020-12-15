// 2020 Marcel Wagenländer

#ifndef ADD_H
#define ADD_H

#include "cuda_helper.hpp"
#include "tensors.hpp"


class Add {
private:
    CudaHelper *cuda_helper_ = NULL;
    Matrix<float> y_;

public:
    Add(CudaHelper *cuda_helper, long num_nodes, long num_features);
    Matrix<float> *op(Matrix<float> *a, Matrix<float> *b);
};

class AddChunked {
private:
    CudaHelper *cuda_helper_ = NULL;
    std::vector<Matrix<float>> y_;

public:
    AddChunked(CudaHelper *cuda_helper, long num_nodes, long chunk_size);
    std::vector<Matrix<float>> *op(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b);
};

#endif//ADD_H
