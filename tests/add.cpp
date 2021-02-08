// Copyright 2020 Marcel Wagenländer

#include "add.hpp"
#include "cuda_helper.hpp"

#include "catch2/catch.hpp"


int test_add_chunked(AddChunked *add, bool keep_allocation) {
    long num_chunks = 64;
    long chunk_size = 32;
    long num_features = 16;
    std::vector<Matrix<float>> a(num_chunks);
    std::vector<Matrix<float>> b(num_chunks);

    for (long i = 0; i < num_chunks; ++i) {
        a.at(i).set(chunk_size, num_features, false);
        b.at(i).set(chunk_size, num_features, false);
        for (long j = 0; j < chunk_size * num_features; ++j) {
            a.at(i).values_[j] = j + 1;
            b.at(i).values_[j] = j + 1;
        }
    }

    CudaHelper cuda_helper;
    add->set(&cuda_helper, chunk_size, chunk_size * num_chunks, num_features, keep_allocation);

    std::vector<Matrix<float>> *y = add->forward(&a, &b);

    for (long i = 0; i < num_chunks; ++i) {
        for (long j = 0; j < chunk_size * num_features; ++j) {
            if (y->at(i).values_[j] != 2 * (j + 1)) {
                return 0;
            }
        }
    }

    return 1;
}

TEST_CASE("Add, chunked", "[add][chunked]") {
    AddChunked add;
    CHECK(test_add_chunked(&add, false));
}

TEST_CASE("Add, chunked, keep", "[add][chunked][keep]") {
    AddChunked add;
    CHECK(test_add_chunked(&add, false));
}

TEST_CASE("Add, pipelined", "[add][pipelined]") {
    AddPipelined add;
    CHECK(test_add_chunked(&add, false));
}

TEST_CASE("Add, pipelined, keep", "[add][pipelined][keep]") {
    AddPipelined add;
    CHECK(test_add_chunked(&add, true));
}
