// 2020 Marcel Wagenländer

#include "add.hpp"


Add::Add(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    cuda_helper_ = cuda_helper;
    y_ = Matrix<float>(num_nodes, num_features, true);
}

Matrix<float>* Add::forward(Matrix<float> *a, Matrix<float> *b) {
    if (a->row_major != b->row_major) {
        to_row_major_inplace(a);
        to_row_major_inplace(b);
    }
    float alpha = 1.0;

    float *d_a;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_a),
                          a->rows * a->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_a, a->values,
                          a->rows * a->columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_b;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_b),
                          a->rows * a->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_b, a->values,
                          a->rows * a->columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                             a->rows * a->columns,
                             &alpha, d_a, 1,
                             d_b, 1));

    check_cuda(cudaMemcpy(y_.values, d_b,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.row_major = a->row_major;

    // clean-up
    check_cuda(cudaFree(d_a));
    check_cuda(cudaFree(d_b));

    return &y_;
}
