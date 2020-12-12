// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "divmv.h"
#include "sparse_computation.hpp"

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
        sp_mat_sum_rows(cuda_helper_, adjacency_, &sum_);
    }
}

Matrix<float> *GraphConvolution::forward(Matrix<float> *x) {
    to_column_major_inplace(x);

    sp_mat_mat_multi(cuda_helper_, adjacency_, x, &y_, false);

    // apply mean
    if (mean_) {
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
    sp_mat_mat_multi(cuda_helper_, adjacency_, &gradients_, &gradients_, false);

    return &gradients_;
}

GraphConvChunked::GraphConvChunked(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                                   long num_features, long chunk_size, long num_nodes) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_nodes_ = num_nodes;
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

    adjacencies_ = std::vector<SparseMatrix<float>>(num_chunks_ * num_chunks_);
    long current_end_row; // end row is included [start_row, end_row] not [start_row, end_row)
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_end_row = i * chunk_size + last_chunk_size_ - 1;
        } else {
            current_end_row = (i + 1) * chunk_size - 1;
        }

        // chunk by row
        SparseMatrix<float> adjacency_chunk;
        get_rows(&adjacency_chunk, adjacency, i * chunk_size, current_end_row);
        // transpose
        transpose_csr_matrix_cpu(&adjacency_chunk);
        // chunk by row (would be by column without transpose
        for (int j = 0; j < num_chunks_; ++j) {
            if (j == num_chunks_ - 1) {
                current_end_row = j * chunk_size + last_chunk_size_ - 1;
            } else {
                current_end_row = (j + 1) * chunk_size - 1;
            }

            get_rows(&adjacencies_[i * num_chunks_ + j], &adjacency_chunk, j * chunk_size, current_end_row);
            // transpose
            transpose_csr_matrix_cpu(&adjacencies_[i * num_chunks_ + j]);
        }
    }

    y_ = std::vector<Matrix<float>>(num_chunks_);
    gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }

        y_.at(i).set(current_chunk_size, num_features, false);
        gradients_.at(i).set(current_chunk_size, num_features, false);
    }

    sum_.set(num_nodes, 1, true);
    if (mean_) {
        sp_mat_sum_rows(adjacency_, &sum_);
    }
}

std::vector<Matrix<float>> *GraphConvChunked::forward(std::vector<Matrix<float>> *x) {
    for (int i = 0; i < x->size(); ++i) {
        to_column_major_inplace(&x->at(i));
    }

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.at(0).size_ * sizeof(float)));

    float *d_sum;
    if (mean_) {
        check_cuda(cudaMalloc(&d_sum, y_.at(0).num_rows_ * sizeof(float)));
    }

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->at(0).size_ * sizeof(float)));

    // row chunk
    for (int i = 0; i < num_chunks_; ++i) {
        // column chunk of row chunk
        check_cuda(cudaMemset(d_y, 0, y_.at(i).size_ * sizeof(float)));

        for (int j = 0; j < num_chunks_; ++j) {
            SparseMatrixCuda<float> d_adj_i;
            malloc_memcpy_sp_mat(&d_adj_i, &adjacencies_[i * num_chunks_ + j]);

            check_cuda(cudaMemcpy(d_x, x->at(j).values_, x->at(j).size_ * sizeof(float), cudaMemcpyHostToDevice));

            sp_mat_mat_multi_cuda(cuda_helper_, &d_adj_i, d_x, d_y, x->at(j).num_columns_, true);
        }

        if (mean_) {
            check_cuda(cudaMemcpy(d_sum, &sum_.values_[i * chunk_size_], y_.at(i).num_rows_ * sizeof(float),
                                  cudaMemcpyHostToDevice));

            div_mat_vec(d_y, d_sum, y_.at(i).num_rows_, y_.at(i).num_columns_);
        }

        check_cuda(cudaMemcpy(y_.at(i).values_, d_y, y_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // free GPU memory
    if (mean_) {
        check_cuda(cudaFree(d_sum));
    }
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_x));

    return &y_;
}

std::vector<Matrix<float>> *GraphConvChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    for (int i = 0; i < incoming_gradients->size(); ++i) {
        to_column_major_inplace(&incoming_gradients->at(i));
    }

    float *d_gradients;
    check_cuda(cudaMalloc(&d_gradients, gradients_.at(0).size_ * sizeof(float)));

    float *d_sum;
    if (mean_) {
        check_cuda(cudaMalloc(&d_sum, incoming_gradients->at(0).num_rows_ * sizeof(float)));
    }

    float *d_incoming_gradients;
    check_cuda(cudaMalloc(&d_incoming_gradients, incoming_gradients->at(0).size_ * sizeof(float)));

    // row chunk
    for (int i = 0; i < num_chunks_; ++i) {
        // column chunk of row chunk
        check_cuda(cudaMemset(d_gradients, 0, gradients_.at(i).size_ * sizeof(float)));

        for (int j = 0; j < num_chunks_; ++j) {
            SparseMatrixCuda<float> d_adj_i;
            malloc_memcpy_sp_mat(&d_adj_i, &adjacencies_[i * num_chunks_ + j]);

            check_cuda(cudaMemcpy(d_incoming_gradients, incoming_gradients->at(j).values_, incoming_gradients->at(j).size_ * sizeof(float), cudaMemcpyHostToDevice));

            if (mean_) {
                check_cuda(cudaMemcpy(d_sum, &sum_.values_[j * chunk_size_], incoming_gradients->at(j).num_rows_ * sizeof(float),
                                      cudaMemcpyHostToDevice));

                div_mat_vec(d_incoming_gradients, d_sum, incoming_gradients->at(j).num_rows_, incoming_gradients->at(j).num_columns_);
            }

            sp_mat_mat_multi_cuda(cuda_helper_, &d_adj_i, d_incoming_gradients, d_gradients, incoming_gradients->at(j).num_columns_, true);
        }

        check_cuda(cudaMemcpy(gradients_.at(i).values_, d_gradients, gradients_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // free GPU memory
    if (mean_) {
        check_cuda(cudaFree(d_sum));
    }
    check_cuda(cudaFree(d_incoming_gradients));
    check_cuda(cudaFree(d_gradients));

    return &gradients_;
}
