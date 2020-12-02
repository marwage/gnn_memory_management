// 2020 Marcel Wagenländer

#include "add.hpp"
#include "dense_computation.hpp"


Add::Add(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    cuda_helper_ = cuda_helper;
    y_.set(num_nodes, num_features, true);
}

Matrix<float> *Add::forward(Matrix<float> *a, Matrix<float> *b) {
    mat_mat_add(cuda_helper_, a, b, &y_);

    return &y_;
}
