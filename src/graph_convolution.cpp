// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "cuda_helper.hpp"
#include "divmv.h"
#include "memory.hpp"

#include <cmath>
#include <string>

const int LOG_MEMORY = 1;


GraphConvolution::GraphConvolution() {}

GraphConvolution::GraphConvolution(CudaHelper *helper, sparse_matrix<float> *adjacency, std::string reduction,
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

    y_ = new_float_matrix(num_nodes, num_features, false);
    gradients_ = new_float_matrix(num_nodes, num_features, false);

    if (mean_) {
        sum_ = new_float_matrix(num_nodes, 1, false);
        ones_ = new_float_matrix(adjacency->columns, 1, false); // adjacency_->columns is chunk_size
        for (int i = 0; i < ones_.rows * ones_.columns; ++i) {
            ones_.values[i] = 1.0;
        }
    }
}

matrix<float>* GraphConvolution::forward(matrix<float> *x) {
    to_column_major_inplace(x);

    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc((void **) &d_A_csr_val,
                          adjacency_->nnz * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_A_csr_row_offsets,
                          (adjacency_->rows + 1) * sizeof(int)));
    check_cuda(cudaMalloc((void **) &d_A_col_ind,
                          adjacency_->nnz * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, adjacency_->csr_val,
                          adjacency_->nnz * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, adjacency_->csr_row_ptr,
                          (adjacency_->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, adjacency_->csr_col_ind,
                          adjacency_->nnz * sizeof(int), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t A_descr;
    check_cusparse(cusparseCreateCsr(&A_descr, adjacency_->rows,
                                     adjacency_->columns, adjacency_->nnz,
                                     d_A_csr_row_offsets, d_A_col_ind,
                                     d_A_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // create cusparse d_x
    float *d_x;
    check_cuda(cudaMalloc((void **) &d_x, x->rows * x->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values, x->rows * x->columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnMatDescr_t x_descr;
    check_cusparse(cusparseCreateDnMat(&x_descr, x->rows, x->columns,
                                       x->rows, d_x,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // create cusparse d_y
    for (int i = 0; i < y_.rows * y_.columns; ++i) {
        y_.values[i] = 0.0f;
    }
    float *d_y;
    check_cuda(cudaMalloc((void **) &d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values, y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cusparseDnMatDescr_t result_descr;
    check_cusparse(cusparseCreateDnMat(&result_descr, y_.rows, y_.columns,
                                       y_.rows, d_y,
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
        for (int i = 0; i < sum_.rows * sum_.columns; ++i) {
            sum_.values[i] = 0.0;
        }

        float *d_ones;
        check_cuda(cudaMalloc(&d_ones, ones_.rows * ones_.columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_ones, ones_.values, ones_.rows * ones_.columns * sizeof(float),
                              cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t ones_desc;
        check_cusparse(cusparseCreateDnVec(&ones_desc, ones_.rows * ones_.columns,
                                           d_ones, CUDA_R_32F));

        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.rows * sum_.columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values, sum_.rows * sum_.columns * sizeof(float),
                              cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t sum_desc;
        check_cusparse(cusparseCreateDnVec(&sum_desc, sum_.rows * sum_.columns,
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

        check_cuda(cudaMemcpy(sum_.values, d_sum,
                              sum_.rows * sum_.columns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        div_mat_vec(d_y, d_sum, y_.rows, y_.columns);

        // free GPU memory
        check_cuda(cudaFree(d_ones));
        check_cuda(cudaFree(d_sum));
    }// end mean

    // copy y_ to CPU memory
    check_cuda(cudaMemcpy(y_.values, d_y,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.row_major = false;

    if (LOG_MEMORY) {
        log_allocated_memory("Graph convolution forward");
    }

    // free memory
    check_cuda(cudaFree(d_A_csr_val));
    check_cuda(cudaFree(d_A_col_ind));
    check_cuda(cudaFree(d_A_csr_row_offsets));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_y));

    return &y_;
}

matrix<float>* GraphConvolution::forward(sparse_matrix<float> *adj, matrix<float> *x) {
    adjacency_ = adj;
    return forward(x);
}

matrix<float>* GraphConvolution::backward(matrix<float> *in_gradients) {
    to_column_major_inplace(in_gradients);

    float *d_g;
    check_cuda(cudaMalloc(&d_g, in_gradients->rows * in_gradients->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_g, in_gradients->values, in_gradients->rows * in_gradients->columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (mean_) {
        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.rows * sum_.columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values, sum_.rows * sum_.columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        div_mat_vec(d_g, d_sum, in_gradients->rows, in_gradients->columns);

        check_cuda(cudaFree(d_sum));
    }

    // gradients_ = adjacency.T * in_gradients
    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    check_cuda(cudaMalloc((void **) &d_A_csr_val,
                          adjacency_->nnz * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_A_csr_row_offsets,
                          (adjacency_->rows + 1) * sizeof(int)));
    check_cuda(cudaMalloc((void **) &d_A_col_ind,
                          adjacency_->nnz * sizeof(int)));
    check_cuda(cudaMemcpy(d_A_csr_val, adjacency_->csr_val,
                          adjacency_->nnz * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_csr_row_offsets, adjacency_->csr_row_ptr,
                          (adjacency_->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_A_col_ind, adjacency_->csr_col_ind,
                          adjacency_->nnz * sizeof(int), cudaMemcpyHostToDevice));
    cusparseSpMatDescr_t A_desc;
    check_cusparse(cusparseCreateCsr(&A_desc, adjacency_->rows,
                                     adjacency_->columns, adjacency_->nnz,
                                     d_A_csr_row_offsets, d_A_col_ind,
                                     d_A_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnMatDescr_t g_desc;
    check_cusparse(cusparseCreateDnMat(&g_desc, in_gradients->rows, in_gradients->columns,
                                       in_gradients->rows, d_g,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    float *d_dinput;
    check_cuda(cudaMalloc((void **) &d_dinput,
                          gradients_.rows * gradients_.columns * sizeof(float)));
    cusparseDnMatDescr_t dinput_desc;
    check_cusparse(cusparseCreateDnMat(&dinput_desc, gradients_.rows, gradients_.columns,
                                       gradients_.rows, d_dinput,
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

    check_cuda(cudaMemcpy(gradients_.values, d_dinput,
                          gradients_.rows * gradients_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.row_major = false;

    if (LOG_MEMORY) {
        log_allocated_memory("Graph convolution backward");
    }

    // clean-up
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_dinput));
    check_cuda(cudaFree(d_A_csr_val));
    check_cuda(cudaFree(d_A_csr_row_offsets));
    check_cuda(cudaFree(d_A_col_ind));
    check_cuda(cudaFree(d_g));

    return &gradients_;
}

GraphConvChunked::GraphConvChunked(CudaHelper *helper, sparse_matrix<float> *adjacency, std::string reduction,
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
    adjacencies_ = std::vector<sparse_matrix<float>>(num_chunks_);
    x_chunks_ = std::vector<matrix<float>>(num_chunks_);
    in_gradients_chunks_ = std::vector<matrix<float>>(num_chunks_);
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            // ONLY POSSIBLE IF ADJACENCY IS SYMMETRIC
            adjacencies_[i] = get_rows(adjacency, i * chunk_size, i * chunk_size + last_chunk_size_);
            transpose_csr_matrix(&adjacencies_[i], cuda_helper_);
            x_chunks_[i].rows = last_chunk_size_;
            in_gradients_chunks_[i].rows = last_chunk_size_;
        } else {
            // ONLY POSSIBLE IF ADJACENCY IS SYMMETRIC
            adjacencies_[i] = get_rows(adjacency, i * chunk_size, (i + 1) * chunk_size);
            transpose_csr_matrix(&adjacencies_[i], cuda_helper_);
            x_chunks_[i].rows = chunk_size_;
            in_gradients_chunks_[i].rows = chunk_size_;
        }
        graph_conv_layers_[i] = GraphConvolution(cuda_helper_, &adjacencies_[i], reduction, num_nodes, num_features);
        x_chunks_[i].columns = num_features;
        in_gradients_chunks_[i].columns = num_features;
        x_chunks_[i].values = new float[x_chunks_[i].rows * x_chunks_[i].columns];
        in_gradients_chunks_[i].values = new float[in_gradients_chunks_[i].rows * in_gradients_chunks_[i].columns];
    }

    y_ = new_float_matrix(num_nodes, num_features, true);
    gradients_ = new_float_matrix(num_nodes, num_features, true);
}

matrix<float>* GraphConvChunked::forward(matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.rows != x->rows || y_.columns != x->columns) {
        throw "Matrix shapes are unequal";
    }

    matrix<float> *y_chunk;

    for (int i = 0; i < y_.rows * y_.columns; ++i) {
        y_.values[i] = 0.0;
    }
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values, y_.rows * y_.columns * sizeof(float),
               cudaMemcpyHostToDevice));

    float *d_y_chunk;
    check_cuda(cudaMalloc(&d_y_chunk, y_.rows * y_.columns * sizeof(float)));

    for (int i = 0; i < num_chunks_; ++i) {
        std::memcpy(x_chunks_[i].values, &x->values[i * chunk_size_ * x->columns],
                    x_chunks_[i].rows * x_chunks_[i].columns * sizeof(float));

        y_chunk = graph_conv_layers_[i].forward(&x_chunks_[i]);

        check_cuda(cudaMemcpy(d_y_chunk, y_chunk->values, y_chunk->rows * y_chunk->columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = 1.0;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 y_.rows * y_.columns,
                                 &alpha,
                                 d_y_chunk, 1,
                                 d_y, 1));
    }

    check_cuda(cudaMemcpy(y_.values, d_y, y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.row_major = y_chunk->row_major;

    if (LOG_MEMORY) {
        log_allocated_memory("Graph convolution chunked forward");
    }

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_y_chunk));


    return &y_;
}

matrix<float>* GraphConvChunked::backward(matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    if (y_.rows != in_gradients->rows || y_.columns != in_gradients->columns) {
        throw "Matrix shapes are unequal";
    }

    matrix<float> *gradients_chunk;
    for (int i = 0; i < gradients_.rows * gradients_.columns; ++i) {
        gradients_.values[i] = 0.0;
    }
    float *d_gradients;
    check_cuda(cudaMalloc(&d_gradients, gradients_.rows * gradients_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_gradients, gradients_.values, gradients_.rows * gradients_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_gradients_chunk;
    check_cuda(cudaMalloc(&d_gradients_chunk, gradients_.rows * gradients_.columns * sizeof(float)));

    for (int i = 0; i < num_chunks_; ++i) {
        std::memcpy(in_gradients_chunks_[i].values, &in_gradients->values[i * chunk_size_ * in_gradients->columns],
                    in_gradients_chunks_[i].rows * in_gradients_chunks_[i].columns * sizeof(float));

        gradients_chunk = graph_conv_layers_[i].backward(&in_gradients_chunks_[i]);

        check_cuda(cudaMemcpy(d_gradients_chunk, gradients_chunk->values, gradients_chunk->rows * gradients_chunk->columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = 1.0;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 gradients_.rows * gradients_.columns,
                                 &alpha,
                                 d_gradients_chunk, 1,
                                 d_gradients, 1));
    }

    check_cuda(cudaMemcpy(gradients_.values, d_gradients, gradients_.rows * gradients_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.row_major = gradients_chunk->row_major;

    if (LOG_MEMORY) {
        log_allocated_memory("Graph convolution chunked backward");
    }

    // free GPU memory
    check_cuda(cudaFree(d_gradients));
    check_cuda(cudaFree(d_gradients_chunk));

    return &gradients_;
}
