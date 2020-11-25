// 2020 Marcel Wagenländer

#ifndef ADD_H
#define ADD_H

#include "tensors.hpp"
#include "cuda_helper.hpp"


class Add {
private:
    CudaHelper *cuda_helper_;
    matrix<float> y_;
public:
    Add(CudaHelper *cuda_helper, long num_nodes, long num_features);
    matrix<float>* forward(matrix<float> *a, matrix<float> *b);
};


#endif//ADD_H
