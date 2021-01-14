// Copyright 2020 Marcel Wagenländer

#include "chunking.hpp"
#include "sparse_computation.hpp"

#include <thread>


void init_set_random_values(std::vector<Matrix<float>> *mat, long num_nodes, long num_features, long chunk_size, bool is_row_major) {
    long num_chunks = mat->size();
    long last_chunk_size;
    if (num_chunks * chunk_size > num_nodes) {
        last_chunk_size = num_nodes - (num_chunks - 1) * chunk_size;
    } else {
        last_chunk_size = chunk_size;
    }
    long current_chunk_size = chunk_size;
    for (long i = 0; i < num_chunks; ++i) {
        if (i == num_chunks - 1) {
            current_chunk_size = last_chunk_size;
        }
        mat->at(i).set(current_chunk_size, num_features, is_row_major);
        mat->at(i).set_random_values();
    }
}

void create_chunk(Matrix<float> *x, std::vector<Matrix<float>> *x_chunked, long i,
                  long chunk_size, long current_chunk_size, long num_features) {
    x_chunked->at(i).set(current_chunk_size, num_features, true);
    std::copy(x->values_ + (i * chunk_size * num_features),
              x->values_ + (i * chunk_size * num_features) + current_chunk_size * num_features,
              x_chunked->at(i).values_);
}

void chunk_up(Matrix<float> *x, std::vector<Matrix<float>> *x_chunked, long chunk_size) {
    to_row_major_inplace(x);

    long num_nodes = x->num_rows_;
    long num_features = x->num_columns_;
    long num_chunks = x_chunked->size();
    long last_chunk_size = 0;
    if (num_chunks * chunk_size > num_nodes) {
        last_chunk_size = num_nodes - (num_chunks - 1) * chunk_size;
    } else {
        last_chunk_size = chunk_size;
    }

    long current_chunk_size = chunk_size;
    std::vector<std::thread> threads(num_chunks);
    for (long i = 0; i < num_chunks; ++i) {
        if (i == num_chunks - 1) {
            current_chunk_size = last_chunk_size;
        }
        threads.at(i) = std::thread(create_chunk, x, x_chunked, i, chunk_size, current_chunk_size, num_features);
    }
    for (long i = 0; i < num_chunks; ++i) {
        threads.at(i).join();
    }
}

void stitch(std::vector<Matrix<float>> *x_chunked, Matrix<float> *x) {
    long num_chunks = x_chunked->size();
    long chunk_size = x_chunked->at(0).num_rows_;
    long num_features = x_chunked->at(0).num_columns_;
    for (int i = 0; i < num_chunks; ++i) {
        to_row_major_inplace(&x_chunked->at(i));
        std::copy(x_chunked->at(i).values_,
                  x_chunked->at(i).values_ + x_chunked->at(i).size_,
                  x->values_ + (i * chunk_size * num_features));
    }
    x->is_row_major_ = true;
}

void double_chunk_up_sp(SparseMatrix<float> *sp_mat, std::vector<SparseMatrix<float>> *chunks, long chunk_size) {
    long num_nodes = sp_mat->num_rows_;
    long num_chunks = ceil((double) num_nodes / (double) chunk_size);
    if ((long) chunks->size() != num_chunks * num_chunks) {
        throw "Vector has wrong number of chunks.";
    }

    long last_chunk_size;
    if (num_chunks * chunk_size > num_nodes) {
        last_chunk_size = num_nodes - (num_chunks - 1) * chunk_size;
    } else {
        last_chunk_size = chunk_size;
    }
    long current_end_row;// end row is included [start_row, end_row] not [start_row, end_row)
    for (int i = 0; i < num_chunks; ++i) {
        if (i == num_chunks - 1) {
            current_end_row = i * chunk_size + last_chunk_size - 1;
        } else {
            current_end_row = (i + 1) * chunk_size - 1;
        }

        // chunk by row
        SparseMatrix<float> sp_mat_chunk;
        get_rows(&sp_mat_chunk, sp_mat, i * chunk_size, current_end_row);
        // transpose
        transpose_csr_matrix_cpu(&sp_mat_chunk);
        // chunk by row (would be by column without transpose
        for (int j = 0; j < num_chunks; ++j) {
            if (j == num_chunks - 1) {
                current_end_row = j * chunk_size + last_chunk_size - 1;
            } else {
                current_end_row = (j + 1) * chunk_size - 1;
            }

            get_rows(&chunks->at(i * num_chunks + j), &sp_mat_chunk, j * chunk_size, current_end_row);
            // transpose
            transpose_csr_matrix_cpu(&chunks->at(i * num_chunks + j));
        }
    }
}
