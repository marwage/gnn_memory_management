// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"

#include "cuda_helper.hpp"

void div_mat_vec(float *X, float *y, int n, int m);


GraphConvolution::GraphConvolution(CudaHelper *helper) {
    cuda_helper_ = helper;
}

matrix<float> GraphConvolution::forward(sparse_matrix<float> A, matrix<float> B,
                                        std::string reduction) {
    bool mean;
    if (reduction.compare("mean") == 0) {
        mean = true;
    } else if (reduction.compare("sum") == 0) {
        mean = false;
    } else {
        std::cout << "Reduction not supported" << std::endl;
    }

    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc((void **) &d_A_csr_val,
                          A.nnz * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_A_csr_row_offsets,
                          (A.rows + 1) * sizeof(int)));
    check_cuda(cudaMalloc((void **) &d_A_col_ind,
                          A.nnz * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, A.csr_val,
                          A.nnz * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, A.csr_row_ptr,
                          (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, A.csr_col_ind,
                          A.nnz * sizeof(int), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t A_descr;
    check_cusparse(cusparseCreateCsr(&A_descr, A.rows,
                                     A.columns, A.nnz,
                                     d_A_csr_row_offsets, d_A_col_ind,
                                     d_A_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // create cusparse B
    float *d_B;
    check_cuda(cudaMalloc((void **) &d_B, B.rows * B.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_B, B.values, B.rows * B.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnMatDescr_t B_descr;
    check_cusparse(cusparseCreateDnMat(&B_descr, B.rows, B.columns,
                                       B.rows, d_B,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // create result
    matrix<float> result;
    result.rows = A.rows;
    result.columns = B.columns;
    result.values = (float *) malloc(result.rows * result.columns * sizeof(float));
    for (int i = 0; i < result.rows * result.columns; ++i) {
        result.values[i] = 0.0f;
    }

    // create cusparse result
    float *d_result;
    check_cuda(cudaMalloc((void **) &d_result, result.rows * result.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_result, result.values, result.rows * result.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnMatDescr_t result_descr;
    check_cusparse(cusparseCreateDnMat(&result_descr, result.rows, result.columns,
                                       result.rows, d_result,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // get buffer size for SpMM
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t buffer_size;
    check_cusparse(cusparseSpMM_bufferSize(cuda_helper_->cusparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A_descr, B_descr, &beta, result_descr,
            // CUSPARSE_MM_ALG_DEFAULT is deprecated
            // but CUSPARSE_SPMM_ALG_DEFAULT is not working
                                           CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                           &buffer_size));
    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    // compute SpMM
    check_cusparse(cusparseSpMM(cuda_helper_->cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A_descr, B_descr, &beta, result_descr,
                                CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                d_buffer));

    check_cuda(cudaFree(d_buffer));

    // apply mean
    if (mean) {
        matrix<float> ones;
        ones.rows = A.rows;
        ones.columns = 1;
        ones.values = (float *) malloc(ones.rows * ones.columns * sizeof(float));
        for (int i = 0; i < ones.rows * ones.columns; ++i) {
            ones.values[i] = 1.0;
        }
        float *d_ones;
        check_cuda(cudaMalloc(&d_ones, ones.rows * ones.columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_ones, ones.values, ones.rows * ones.columns * sizeof(float),
                              cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t ones_desc;
        check_cusparse(cusparseCreateDnVec(&ones_desc, ones.rows,
                                           d_ones, CUDA_R_32F));

        matrix<float> sum;
        sum.rows = ones.rows;
        sum.columns = 1;
        sum.values = (float *) malloc(sum.rows * sum.columns * sizeof(float));
        for (int i = 0; i < sum.rows * sum.columns; ++i) {
            sum.values[0] = 0.0;
        }
        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum.rows * sum.columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum.values, sum.rows * sum.columns * sizeof(float),
                              cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t sum_desc;
        check_cusparse(cusparseCreateDnVec(&sum_desc, sum.rows * sum.columns,
                                           d_sum, CUDA_R_32F));

        check_cusparse(cusparseSpMV_bufferSize(cuda_helper_->cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, A_descr, ones_desc,
                                               &beta, sum_desc,
                                               CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer_size));
        check_cuda(cudaMalloc(&d_buffer, buffer_size));
        check_cusparse(cusparseSpMV(cuda_helper_->cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, A_descr, ones_desc,
                                    &beta, sum_desc,
                                    CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, d_buffer));

        check_cuda(cudaMemcpy(sum.values, d_sum,
                              sum.rows * sum.columns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        div_mat_vec(d_result, d_sum, result.rows, result.columns);

        // free GPU memory
        check_cuda(cudaFree(d_ones));
        check_cuda(cudaFree(d_sum));

        // free CPU memory
        free(ones.values);
        free(sum.values);
    }  // end mean

    // copy result to CPU memory
    check_cuda(cudaMemcpy(result.values, d_result,
                          result.rows * result.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free memory
    check_cuda(cudaFree(d_A_csr_val));
    check_cuda(cudaFree(d_A_col_ind));
    check_cuda(cudaFree(d_A_csr_row_offsets));
    check_cuda(cudaFree(d_B));
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_result));

    return result;
}
