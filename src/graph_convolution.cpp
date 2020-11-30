// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "cuda_helper.hpp"
#include "divmv.h"

#include <cmath>
#include <string>


GraphConvolution::GraphConvolution() {}

GraphConvolution::GraphConvolution(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                                   long num_nodes, long num_features) {
    set(helper, adjacency, reduction, num_nodes, num_features);
}

void GraphConvolution::set(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                           long num_nodes, long num_features) {
    cuda_helper_ = helper;
    adjacency_ = adjacency;
    reduction_ = reduction;
    if (reduction_.compare("mean") == 0) {
        mean_ = true;
    } else if (reduction_.compare("sum") == 0) {
        mean_ = false;
    } else {
        throw "Reduction not supported";
    }

    y_.set(num_nodes, num_features, false);
    gradients_.set(num_nodes, num_features, false);

    if (mean_) {
        sum_.set(num_nodes, 1, false);
        ones_.set(adjacency->num_columns_, 1, false);// adjacency_->columns is chunk_size
        for (int i = 0; i < ones_.size_; ++i) {
            ones_.values_[i] = 1.0;
        }
    }
}

Matrix<float> *GraphConvolution::forward(Matrix<float> *x) {
    to_column_major_inplace(x);

    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc((void **) &d_A_csr_val,
                          adjacency_->nnz_ * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_A_csr_row_offsets,
                          (adjacency_->num_rows_ + 1) * sizeof(int)));
    check_cuda(cudaMalloc((void **) &d_A_col_ind,
                          adjacency_->nnz_ * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, adjacency_->csr_val_,
                          adjacency_->nnz_ * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, adjacency_->csr_row_ptr_,
                          (adjacency_->num_rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, adjacency_->csr_col_ind_,
                          adjacency_->nnz_ * sizeof(int), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t A_descr;
    check_cusparse(cusparseCreateCsr(&A_descr, adjacency_->num_rows_,
                                     adjacency_->num_columns_, adjacency_->nnz_,
                                     d_A_csr_row_offsets, d_A_col_ind,
                                     d_A_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // create cusparse d_x
    float *d_x;
    check_cuda(cudaMalloc((void **) &d_x, x->num_rows_ * x->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->num_rows_ * x->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnMatDescr_t x_descr;
    check_cusparse(cusparseCreateDnMat(&x_descr, x->num_rows_, x->num_columns_,
                                       x->num_rows_, d_x,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // create cusparse d_y
    for (long i = 0; i < y_.size_; ++i) {
        y_.values_[i] = 0.0;
    }
    float *d_y;
    check_cuda(cudaMalloc((void **) &d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnMatDescr_t result_descr;
    check_cusparse(cusparseCreateDnMat(&result_descr, y_.num_rows_, y_.num_columns_,
                                       y_.num_rows_, d_y,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // get buffer size for SpMM
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t buffer_size;
    check_cusparse(cusparseSpMM_bufferSize(cuda_helper_->cusparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A_descr, x_descr, &beta, result_descr,
                                           // CUSPARSE_MM_ALG_DEFAULT is deprecated
                                           // but CUSPARSE_SPMM_ALG_DEFAULT is not working
                                           CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                           &buffer_size));
    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    // compute SpMM
    check_cusparse(cusparseSpMM(cuda_helper_->cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A_descr, x_descr, &beta, result_descr,
                                CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                d_buffer));

    check_cuda(cudaFree(d_buffer));

    // apply mean
    if (mean_) {
        for (long i = 0; i < sum_.size_; ++i) {
            sum_.values_[i] = 0.0;
        }

        float *d_ones;
        check_cuda(cudaMalloc(&d_ones, ones_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_ones, ones_.values_, ones_.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t ones_desc;
        check_cusparse(cusparseCreateDnVec(&ones_desc, ones_.size_,
                                           d_ones, CUDA_R_32F));

        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values_, sum_.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t sum_desc;
        check_cusparse(cusparseCreateDnVec(&sum_desc, sum_.size_,
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

        check_cuda(cudaMemcpy(sum_.values_, d_sum,
                              sum_.size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        div_mat_vec(d_y, d_sum, y_.num_rows_, y_.num_columns_);

        // free GPU memory
        check_cuda(cudaFree(d_ones));
        check_cuda(cudaFree(d_sum));
    }// end mean

    // copy y_ to CPU memory
    check_cuda(cudaMemcpy(y_.values_, d_y,
                          y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = false;

    // free memory
    check_cuda(cudaFree(d_A_csr_val));
    check_cuda(cudaFree(d_A_col_ind));
    check_cuda(cudaFree(d_A_csr_row_offsets));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *GraphConvolution::forward(SparseMatrix<float> *adj, Matrix<float> *x) {
    adjacency_ = adj;
    return forward(x);
}

Matrix<float> *GraphConvolution::backward(Matrix<float> *in_gradients) {
    to_column_major_inplace(in_gradients);

    float *d_g;
    check_cuda(cudaMalloc(&d_g, in_gradients->num_rows_ * in_gradients->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_g, in_gradients->values_, in_gradients->num_rows_ * in_gradients->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (mean_) {
        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values_, sum_.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        div_mat_vec(d_g, d_sum, in_gradients->num_rows_, in_gradients->num_columns_);

        check_cuda(cudaFree(d_sum));
    }

    // gradients_ = adjacency.T * in_gradients
    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc((void **) &d_A_csr_val,
                          adjacency_->nnz_ * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_A_csr_row_offsets,
                          (adjacency_->num_rows_ + 1) * sizeof(int)));
    check_cuda(cudaMalloc((void **) &d_A_col_ind,
                          adjacency_->nnz_ * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, adjacency_->csr_val_,
                          adjacency_->nnz_ * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, adjacency_->csr_row_ptr_,
                          (adjacency_->num_rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, adjacency_->csr_col_ind_,
                          adjacency_->nnz_ * sizeof(int), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t A_desc;
    check_cusparse(cusparseCreateCsr(&A_desc, adjacency_->num_rows_,
                                     adjacency_->num_columns_, adjacency_->nnz_,
                                     d_A_csr_row_offsets, d_A_col_ind,
                                     d_A_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnMatDescr_t g_desc;
    check_cusparse(cusparseCreateDnMat(&g_desc, in_gradients->num_rows_, in_gradients->num_columns_,
                                       in_gradients->num_rows_, d_g,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    float *d_dinput;
    check_cuda(cudaMalloc((void **) &d_dinput,
                          gradients_.size_ * sizeof(float)));
    cusparseDnMatDescr_t dinput_desc;
    check_cusparse(cusparseCreateDnMat(&dinput_desc, gradients_.num_rows_, gradients_.num_columns_,
                                       gradients_.num_rows_, d_dinput,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    float alpha = 1.0;
    float beta = 0.0;
    size_t buffer_size;
    check_cusparse(cusparseSpMM_bufferSize(cuda_helper_->cusparse_handle,
                                           CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, A_desc, g_desc, &beta, dinput_desc,
                                           CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                           &buffer_size));
    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    // compute SpMM
    check_cusparse(cusparseSpMM(cuda_helper_->cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, A_desc, g_desc, &beta, dinput_desc,
                                CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                d_buffer));

    check_cuda(cudaMemcpy(gradients_.values_, d_dinput,
                          gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = false;

    // clean-up
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_dinput));
    check_cuda(cudaFree(d_A_csr_val));
    check_cuda(cudaFree(d_A_csr_row_offsets));
    check_cuda(cudaFree(d_A_col_ind));
    check_cuda(cudaFree(d_g));

    return &gradients_;
}

GraphConvChunked::GraphConvChunked(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                                   long num_features, long chunk_size, long num_nodes) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    graph_conv_layers_ = std::vector<GraphConvolution>(num_chunks_);
    adjacencies_ = std::vector<SparseMatrix<float>>(num_chunks_);
    x_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    in_gradients_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    long current_end_row;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_end_row = i * chunk_size + last_chunk_size_;
            current_chunk_size = last_chunk_size_;
        } else {
            current_end_row = (i + 1) * chunk_size;
        }

        // ONLY POSSIBLE IF ADJACENCY IS SYMMETRIC
        get_rows(&adjacencies_[i], adjacency, i * chunk_size, current_end_row);
        transpose_csr_matrix(&adjacencies_[i], cuda_helper_);

        graph_conv_layers_[i].set(cuda_helper_, &adjacencies_[i], reduction, num_nodes, num_features);
        x_chunks_[i].set(current_chunk_size, num_features, true);
        in_gradients_chunks_[i].set(current_chunk_size, num_features, true);
    }

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *GraphConvChunked::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    Matrix<float> *y_chunk;

    for (long i = 0; i < y_.size_; ++i) {
        y_.values_[i] = 0.0;
    }
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_y_chunk;
    check_cuda(cudaMalloc(&d_y_chunk, y_.size_ * sizeof(float)));

    for (int i = 0; i < num_chunks_; ++i) {
        std::memcpy(x_chunks_[i].values_, &x->values_[i * chunk_size_ * x->num_columns_],
                    x_chunks_[i].num_rows_ * x_chunks_[i].num_columns_ * sizeof(float));

        y_chunk = graph_conv_layers_[i].forward(&x_chunks_[i]);

        check_cuda(cudaMemcpy(d_y_chunk, y_chunk->values_, y_chunk->num_rows_ * y_chunk->num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = 1.0;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 y_.size_,
                                 &alpha,
                                 d_y_chunk, 1,
                                 d_y, 1));
    }

    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = y_chunk->is_row_major_;

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_y_chunk));

    return &y_;
}

Matrix<float> *GraphConvChunked::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    if (y_.num_rows_ != in_gradients->num_rows_ || y_.num_columns_ != in_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    for (long i = 0; i < gradients_.size_; ++i) {
        gradients_.values_[i] = 0.0;
    }
    Matrix<float> *gradients_chunk;
    float *d_gradients;
    check_cuda(cudaMalloc(&d_gradients, gradients_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_gradients, gradients_.values_, gradients_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_gradients_chunk;
    check_cuda(cudaMalloc(&d_gradients_chunk, gradients_.size_ * sizeof(float)));

    for (int i = 0; i < num_chunks_; ++i) {
        std::memcpy(in_gradients_chunks_[i].values_, &in_gradients->values_[i * chunk_size_ * in_gradients->num_columns_],
                    in_gradients_chunks_[i].num_rows_ * in_gradients_chunks_[i].num_columns_ * sizeof(float));

        gradients_chunk = graph_conv_layers_[i].backward(&in_gradients_chunks_[i]);

        check_cuda(cudaMemcpy(d_gradients_chunk, gradients_chunk->values_, gradients_chunk->num_rows_ * gradients_chunk->num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = 1.0;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 gradients_.size_,
                                 &alpha,
                                 d_gradients_chunk, 1,
                                 d_gradients, 1));
    }

    check_cuda(cudaMemcpy(gradients_.values_, d_gradients, gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = gradients_chunk->is_row_major_;

    // free GPU memory
    check_cuda(cudaFree(d_gradients));
    check_cuda(cudaFree(d_gradients_chunk));

    return &gradients_;
}
