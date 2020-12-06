// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "cuda_helper.hpp"
#include "divmv.h"
#include "sparse_computation.hpp"
#include "chunk.hpp"

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
    }
}

Matrix<float> *GraphConvolution::forward(Matrix<float> *x) {
    to_column_major_inplace(x);

    sp_mat_mat_multi(cuda_helper_, adjacency_, x, &y_);

    // apply mean
    if (mean_) {
        sp_mat_sum_rows(cuda_helper_, adjacency_, &sum_);

        float *d_y;
        check_cuda(cudaMalloc((void **) &d_y, y_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values_, sum_.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        div_mat_vec(d_y, d_sum, y_.num_rows_, y_.num_columns_);

        check_cuda(cudaMemcpy(y_.values_, d_y,
                              y_.size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        check_cuda(cudaFree(d_y));
        check_cuda(cudaFree(d_sum));
    }

    y_.is_row_major_ = false;

    return &y_;
}

Matrix<float> *GraphConvolution::backward(Matrix<float> *in_gradients) {
    to_column_major_inplace(in_gradients);

    if (mean_) {
        float *d_g;
        check_cuda(cudaMalloc(&d_g, in_gradients->size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_g, in_gradients->values_,
                              in_gradients->size_ * sizeof(float), cudaMemcpyHostToDevice));

        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values_,
                              sum_.size_ * sizeof(float), cudaMemcpyHostToDevice));

        div_mat_vec(d_g, d_sum, in_gradients->num_rows_, in_gradients->num_columns_);

        check_cuda(cudaMemcpy(gradients_.values_, d_g,
                              gradients_.size_ * sizeof(float), cudaMemcpyDeviceToHost));

        check_cuda(cudaFree(d_g));
        check_cuda(cudaFree(d_sum));
    }

    // gradients_ = adjacency.T * in_gradients
    sp_mat_mat_multi(cuda_helper_, adjacency_, &gradients_, &gradients_);

    return &gradients_;
}

GraphConvChunked::GraphConvChunked(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                                   long num_features, long chunk_size, long num_nodes) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    adjacency_ = adjacency;
    if (reduction.compare("mean") == 0) {
        mean_ = true;
    } else if (reduction.compare("sum") == 0) {
        mean_ = false;
    } else {
        throw "Reduction not supported";
    }

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    adjacencies_ = std::vector<SparseMatrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    long current_end_row = 0;
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
    }

    sum_.set(num_nodes, 1, true);
    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *GraphConvChunked::forward(std::vector<Matrix<float>> *x) {
    for (int i = 0; i < x->size(); ++i) {
        to_column_major_inplace(&x->at(i));
    }

    y_.set_values(0.0);
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    Matrix<float> y_part(y_.num_rows_, y_.num_columns_, false);
    float *d_y_part;
    check_cuda(cudaMalloc(&d_y_part, y_.size_ * sizeof(float)));

    float alpha = 1.0;
    for (int i = 0; i < num_chunks_; ++i) {
        sp_mat_mat_multi(cuda_helper_, &adjacencies_[i], &x->at(i), &y_part);

        check_cuda(cudaMemcpy(d_y_part, y_part.values_,
                              y_part.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 y_.size_,
                                 &alpha,
                                 d_y_part, 1,
                                 d_y, 1));
    }

    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_y_part));

    y_.is_row_major_ = y_part.is_row_major_;

    if (mean_) {
        sp_mat_sum_rows(cuda_helper_, adjacency_, &sum_);

        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values_, sum_.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        div_mat_vec(d_y, d_sum, y_.num_rows_, y_.num_columns_);

        check_cuda(cudaMemcpy(y_.values_, d_y,
                              y_.size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        check_cuda(cudaFree(d_sum));
    }

    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *GraphConvChunked::backward(Matrix<float> *incoming_gradients) {
    if (y_.num_rows_ != incoming_gradients->num_rows_ || y_.num_columns_ != incoming_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    Matrix<float> incoming_gradients_scaled;
    if (mean_) {
        to_column_major_inplace(incoming_gradients);

        float *d_incoming_gradients;
        check_cuda(cudaMalloc(&d_incoming_gradients, incoming_gradients->size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_incoming_gradients, incoming_gradients->values_,
                              incoming_gradients->size_ * sizeof(float), cudaMemcpyHostToDevice));

        float *d_sum;
        check_cuda(cudaMalloc(&d_sum, sum_.size_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_sum, sum_.values_,
                              sum_.size_ * sizeof(float), cudaMemcpyHostToDevice));

        div_mat_vec(d_incoming_gradients, d_sum, incoming_gradients->num_rows_, incoming_gradients->num_columns_);

        // TESTING
        incoming_gradients_scaled.set(incoming_gradients->num_rows_, incoming_gradients->num_columns_,
                                                incoming_gradients->is_row_major_);


        check_cuda(cudaMemcpy(incoming_gradients_scaled.values_, d_incoming_gradients,
                              incoming_gradients_scaled.size_ * sizeof(float), cudaMemcpyDeviceToHost));

        // TESTING
        incoming_gradients = &incoming_gradients_scaled;

        check_cuda(cudaFree(d_incoming_gradients));
        check_cuda(cudaFree(d_sum));
    }

    to_row_major_inplace(incoming_gradients);

    gradients_.set_values(0.0);
    float *d_gradients;
    check_cuda(cudaMalloc(&d_gradients, gradients_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_gradients, gradients_.values_, gradients_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    Matrix<float> gradients_part(incoming_gradients->num_rows_, incoming_gradients->num_columns_, false);
    float *d_gradients_part;
    check_cuda(cudaMalloc(&d_gradients_part, gradients_.size_ * sizeof(float)));

    Matrix<float> incoming_gradients_chunk_row;
    Matrix<float> incoming_gradients_chunk;
    long current_chunk_size = chunk_size_;
    float alpha = 1.0;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }
        incoming_gradients_chunk_row.set(current_chunk_size, incoming_gradients->num_columns_,
                                   &incoming_gradients->values_[i * chunk_size_ * incoming_gradients->num_columns_],
                                   true, false);
        incoming_gradients_chunk.set(current_chunk_size, incoming_gradients->num_columns_, false);
        to_column_major(&incoming_gradients_chunk, &incoming_gradients_chunk_row);

        sp_mat_mat_multi(cuda_helper_, &adjacencies_[i], &incoming_gradients_chunk, &gradients_part);

        check_cuda(cudaMemcpy(d_gradients_part, gradients_part.values_,
                              gradients_part.size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 gradients_.size_,
                                 &alpha,
                                 d_gradients_part, 1,
                                 d_gradients, 1));
    }

    check_cuda(cudaMemcpy(gradients_.values_, d_gradients, gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = false;

    // free GPU memory
    check_cuda(cudaFree(d_gradients));
    check_cuda(cudaFree(d_gradients_part));

    return &gradients_;
}
