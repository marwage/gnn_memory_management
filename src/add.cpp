// 2020 Marcel Wagenländer

#include "add.hpp"


Add::Add(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    cuda_helper_ = cuda_helper;
    y_.set(num_nodes, num_features, true);
}

Matrix<float> *Add::forward(Matrix<float> *a, Matrix<float> *b) {
    if (a->is_row_major_ != b->is_row_major_) {
        to_row_major_inplace(a);
        to_row_major_inplace(b);
    }
    float alpha = 1.0;

    float *d_a;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_a),
                          a->num_rows_ * a->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_a, a->values_,
                          a->num_rows_ * a->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_b;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_b),
                          a->num_rows_ * a->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_b, a->values_,
                          a->num_rows_ * a->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                             a->num_rows_ * a->num_columns_,
                             &alpha, d_a, 1,
                             d_b, 1));

    check_cuda(cudaMemcpy(y_.values_, d_b,
                          y_.num_rows_ * y_.num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = a->is_row_major_;

    // clean-up
    check_cuda(cudaFree(d_a));
    check_cuda(cudaFree(d_b));

    return &y_;
}
