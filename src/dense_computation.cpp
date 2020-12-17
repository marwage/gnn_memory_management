// 2020 Marcel Wagenländer

#include "dense_computation.hpp"

void mat_mat_add_cuda(CudaHelper *cuda_helper, float *d_mat_a, float *d_mat_b, long size) {
    float alpha = 1.0;

    check_cublas(cublasSaxpy(cuda_helper->cublas_handle,
                             size,
                             &alpha, d_mat_a, 1,
                             d_mat_b, 1));
}


void mat_mat_add(CudaHelper *cuda_helper, Matrix<float> *mat_a, Matrix<float> *mat_b, Matrix<float> *result) {
    if (mat_a->is_row_major_ != mat_b->is_row_major_) {
        to_row_major_inplace(mat_a);
        to_row_major_inplace(mat_b);
    }

    float *d_mat_a;
    check_cuda(cudaMalloc(&d_mat_a,
                          mat_a->num_rows_ * mat_a->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_mat_a, mat_a->values_, mat_a->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_mat_b;
    check_cuda(cudaMalloc(&d_mat_b, mat_b->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_mat_b, mat_b->values_, mat_b->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    mat_mat_add_cuda(cuda_helper, d_mat_a, d_mat_b, mat_a->size_);

    check_cuda(cudaMemcpy(result->values_, d_mat_b, result->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    result->is_row_major_ = mat_a->is_row_major_;

    // clean-up
    check_cuda(cudaFree(d_mat_a));
    check_cuda(cudaFree(d_mat_b));
}
