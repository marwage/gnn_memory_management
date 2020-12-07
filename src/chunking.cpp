// Copyright 2020 Marcel Wagenländer

#include "chunking.hpp"


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
    for (int i = 0; i < num_chunks; ++i) {
        if (i == num_chunks - 1) {
            current_chunk_size = last_chunk_size;
        }
        x_chunked->at(i).set(current_chunk_size, num_features, true);
        std::copy(x->values_ + (i * chunk_size * num_features),
                  x->values_ + (i * chunk_size * num_features) + current_chunk_size * num_features,
                  x_chunked->at(i).values_);
    }
}

void stitch(std::vector<Matrix<float>> *x_chunked, Matrix<float> *x) {
    long chunk_size = x_chunked->at(0).num_rows_;
    long num_features = x_chunked->at(0).num_columns_;
    for (int i = 0; i < x_chunked->size(); ++i) {
        to_row_major_inplace(&x_chunked->at(i));
        std::copy(x_chunked->at(i).values_,
                  x_chunked->at(i).values_ + x_chunked->at(i).size_,
                  x->values_ + (i * chunk_size * num_features));
    }
    x->is_row_major_ = true;
}
