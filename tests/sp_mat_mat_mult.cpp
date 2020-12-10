// Copyright 2020 Marcel Wagenländer

#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "sparse_computation.hpp"
#include "tensors.hpp"

#include <catch2/catch.hpp>
#include <vector>


void sp_mat_mat_mult_chunked(CudaHelper *cuda_helper, std::vector<SparseMatrix<float>> *sp, std::vector<Matrix<float>> *x, std::vector<Matrix<float>> *y) {
    long num_chunks = x->size();

    for (int i = 0; i < x->size(); ++i) {
        to_column_major_inplace(&x->at(i));
    }

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y->at(0).size_ * sizeof(float)));

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->at(0).size_ * sizeof(float)));

    // row chunk
    for (int i = 0; i < num_chunks; ++i) {
        // column chunk of row chunk
        check_cuda(cudaMemset(d_y, 0, y->at(i).size_ * sizeof(float)));

        for (int j = 0; j < num_chunks; ++j) {
            SparseMatrixCuda<float> d_adj_i;
            malloc_memcpy_sp_mat(&d_adj_i, &sp->at(i * num_chunks + j));

            check_cuda(cudaMemcpy(d_x, x->at(j).values_, x->at(j).size_ * sizeof(float), cudaMemcpyHostToDevice));

            sp_mat_mat_multi_cuda(cuda_helper, &d_adj_i, d_x, d_y, x->at(j).num_columns_, true);
        }

        check_cuda(cudaMemcpy(y->at(i).values_, d_y, y->at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_x));
}

TEST_CASE("Sparse-matrix matrix multiplication, chunked", "[spmatmatmult][chunked]") {
    CudaHelper cuda_helper;
    long chunk_size = 3;
    long num_nodes = 9;
    long nnz = 8;

    float *values = new float[nnz]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int *col_ind = new int[nnz]{0, 1, 2, 3, 4, 5, 6, 7};
    int *row_ptr = new int[num_nodes + 1]{0, 2, 2, 2, 5, 5, 6, 8, 8, 8};

    SparseMatrix<float> sp_mat(num_nodes, num_nodes, nnz, values, row_ptr, col_ind);

    Matrix<float> x(num_nodes, 3, true);
    for (int i = 0; i < x.size_; ++i) {
        x.values_[i] = i + 1;
    }

    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    long last_chunk_size;
    if (num_chunks * chunk_size > num_nodes) {
        last_chunk_size = num_nodes - (num_chunks - 1) * chunk_size;
    } else {
        last_chunk_size = chunk_size;
    }

    std::vector<SparseMatrix<float>> sp_mat_chunked(num_chunks * num_chunks);
    long current_end_row = 0;
    for (int i = 0; i < num_chunks; ++i) {
        if (i == num_chunks - 1) {
            current_end_row = i * chunk_size + last_chunk_size;
        } else {
            current_end_row = (i + 1) * chunk_size;
        }

        // chunk by row
        SparseMatrix<float> adjacency_chunk;
        get_rows(&adjacency_chunk, &sp_mat, i * chunk_size, current_end_row);
        // transpose
        transpose_csr_matrix(&adjacency_chunk, &cuda_helper);
        // chunk by row (would be by column without transpose)
        for (int j = 0; j < num_chunks; ++j) {
            if (j == num_chunks - 1) {
                current_end_row = j * chunk_size + last_chunk_size;
            } else {
                current_end_row = (j + 1) * chunk_size;
            }

            get_rows(&sp_mat_chunked[i * num_chunks + j], &adjacency_chunk, j * chunk_size, current_end_row);
            // transpose
            transpose_csr_matrix(&sp_mat_chunked[i * num_chunks + j], &cuda_helper);
        }
    }

    std::vector<Matrix<float>> x_chunked(num_chunks);
    chunk_up(&x, &x_chunked, chunk_size);

    std::vector<Matrix<float>> y_chunked(num_chunks);
    for (int i = 0; i < num_chunks; ++i) {
        y_chunked.at(i).set(x_chunked.at(i).num_rows_, x_chunked.at(i).num_columns_, false);
    }

    sp_mat_mat_mult_chunked(&cuda_helper, &sp_mat_chunked, &x_chunked, &y_chunked);

    Matrix<float> y(x.num_rows_, x.num_columns_, true);

    sp_mat_mat_multi(&cuda_helper, &sp_mat, &x, &y, false);

    Matrix<float> y_one(x.num_rows_, x.num_columns_, true);
    stitch(&y_chunked, &y_one);

    CHECK(compare_mat(&y, &y_one, "Aggregation"));
}
