// 2020 Marcel Wagenländer

#include "sparse_computation.hpp"

#include <vector>


void malloc_memcpy_sp_mat(SparseMatrixCuda<float> *d_sp_mat, SparseMatrix<float> *sp_mat) {
    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc(&d_A_csr_val,
                          sp_mat->nnz_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_A_csr_row_offsets,
                          (sp_mat->num_rows_ + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_A_col_ind,
                          sp_mat->nnz_ * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, sp_mat->csr_val_,
                          sp_mat->nnz_ * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, sp_mat->csr_row_ptr_,
                          (sp_mat->num_rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, sp_mat->csr_col_ind_,
                          sp_mat->nnz_ * sizeof(int), cudaMemcpyHostToDevice));

    d_sp_mat->set(sp_mat->num_rows_, sp_mat->num_columns_, sp_mat->nnz_,
                  d_A_csr_val, d_A_csr_row_offsets, d_A_col_ind);
}

void sp_mat_mat_multi(CudaHelper *cuda_helper, SparseMatrix<float> *sp_mat, Matrix<float> *mat, Matrix<float> *result, bool add_to_result) {
    SparseMatrixCuda<float> d_sp_mat;
    malloc_memcpy_sp_mat(&d_sp_mat, sp_mat);

    to_column_major_inplace(mat);
    float *d_mat;
    check_cuda(cudaMalloc(&d_mat, mat->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_mat, mat->values_, mat->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_result;
    check_cuda(cudaMalloc(&d_result, result->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_result, result->values_, result->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    sp_mat_mat_multi_cuda(cuda_helper, &d_sp_mat, d_mat, d_result, mat->num_columns_, add_to_result);

    // copy result to CPU memory
    check_cuda(cudaMemcpy(result->values_, d_result,
                          result->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    result->is_row_major_ = false;

    // free memory
    check_cuda(cudaFree(d_mat));
    check_cuda(cudaFree(d_result));
}

void sp_mat_mat_multi_cuda(CudaHelper *cuda_helper, SparseMatrixCuda<float> *d_sp_mat, float *d_mat, float *d_result, long mat_columns, bool add_to_result) {
    if (d_sp_mat->nnz_ == 0) {
        //        check_cuda(cudaMemset(d_result, 0, d_sp_mat->num_rows_ * mat_columns * sizeof(float)));
        return;
    }

    cusparseSpMatDescr_t a_descr;
    check_cusparse(cusparseCreateCsr(&a_descr, d_sp_mat->num_rows_,
                                     d_sp_mat->num_columns_, d_sp_mat->nnz_,
                                     d_sp_mat->csr_row_ptr_, d_sp_mat->csr_col_ind_,
                                     d_sp_mat->csr_val_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // create cusparse d_mat
    cusparseDnMatDescr_t x_descr;
    check_cusparse(cusparseCreateDnMat(&x_descr, d_sp_mat->num_columns_, mat_columns,
                                       d_sp_mat->num_columns_, d_mat,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // create cusparse d_result
    cusparseDnMatDescr_t result_descr;
    check_cusparse(cusparseCreateDnMat(&result_descr, d_sp_mat->num_rows_, mat_columns,
                                       d_sp_mat->num_rows_, d_result,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // get buffer size for SpMM
    float alpha = 1.0;
    float beta = 0.0;
    if (add_to_result) {
        beta = 1.0;
    }
    size_t buffer_size;
    check_cusparse(cusparseSpMM_bufferSize(cuda_helper->cusparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, a_descr, x_descr, &beta, result_descr,
                                           // CUSPARSE_MM_ALG_DEFAULT is deprecated
                                           // but CUSPARSE_SPMM_ALG_DEFAULT is not working
                                           CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                           &buffer_size));
    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    // compute SpMM
    check_cusparse(cusparseSpMM(cuda_helper->cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, a_descr, x_descr, &beta, result_descr,
                                CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                d_buffer));

    // free memory
    check_cuda(cudaFree(d_buffer));
}

void sp_mat_sum_rows(CudaHelper *cuda_helper, SparseMatrix<float> *sp_mat, Matrix<float> *sum) {
    sum->set_values(0.0);

    std::vector<float> ones(sp_mat->num_columns_, 1.0);

    float *d_ones;
    check_cuda(cudaMalloc(&d_ones, ones.size() * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones, ones.data(), ones.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t ones_desc;
    check_cusparse(cusparseCreateDnVec(&ones_desc, ones.size(),
                                       d_ones, CUDA_R_32F));

    float *d_sum;
    check_cuda(cudaMalloc(&d_sum, sum->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_sum, sum->values_, sum->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t sum_desc;
    check_cusparse(cusparseCreateDnVec(&sum_desc, sum->size_,
                                       d_sum, CUDA_R_32F));

    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc((void **) &d_A_csr_val,
                          sp_mat->nnz_ * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_A_csr_row_offsets,
                          (sp_mat->num_rows_ + 1) * sizeof(int)));
    check_cuda(cudaMalloc((void **) &d_A_col_ind,
                          sp_mat->nnz_ * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, sp_mat->csr_val_,
                          sp_mat->nnz_ * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, sp_mat->csr_row_ptr_,
                          (sp_mat->num_rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, sp_mat->csr_col_ind_,
                          sp_mat->nnz_ * sizeof(int), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t A_descr;
    check_cusparse(cusparseCreateCsr(&A_descr, sp_mat->num_rows_,
                                     sp_mat->num_columns_, sp_mat->nnz_,
                                     d_A_csr_row_offsets, d_A_col_ind,
                                     d_A_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float alpha = 1.0;
    float beta = 0.0;
    size_t buffer_size;
    check_cusparse(cusparseSpMV_bufferSize(cuda_helper->cusparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A_descr, ones_desc,
                                           &beta, sum_desc,
                                           CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer_size));

    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    check_cuda(cudaMalloc(&d_buffer, buffer_size));
    check_cusparse(cusparseSpMV(cuda_helper->cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A_descr, ones_desc,
                                &beta, sum_desc,
                                CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, d_buffer));

    check_cuda(cudaMemcpy(sum->values_, d_sum,
                          sum->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_ones));
    check_cuda(cudaFree(d_sum));
    check_cuda(cudaFree(d_A_csr_val));
    check_cuda(cudaFree(d_A_csr_row_offsets));
    check_cuda(cudaFree(d_A_col_ind));
    check_cuda(cudaFree(d_buffer));
}
