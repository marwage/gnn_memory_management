// Copyright 2020 Marcel Wagenländer

#include "chunking.hpp"

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
