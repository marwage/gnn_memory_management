// 2020 Marcel Wagenländer

#ifndef ADD_H
#define ADD_H

#include "tensors.hpp"
#include "cuda_helper.hpp"


class Add {
private:
    CudaHelper *cuda_helper_;
    Matrix<float> y_;
public:
    Add(CudaHelper *cuda_helper, long num_nodes, long num_features);
    Matrix<float>* forward(Matrix<float> *a, Matrix<float> *b);
};


#endif//ADD_H
