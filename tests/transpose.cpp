// Copyright 2020 Marcel Wagenländer

#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


int test_to_column_major(int rows, int columns) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    matrix<float> mat = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/matrix.npy";
    save_npy_matrix(mat, path);

    matrix<float> mat_transposed = to_column_major(&mat);
    path = test_dir_path + "/matrix_transposed.npy";
    save_npy_matrix(mat_transposed, path);

    to_column_major_inplace(&mat);
    path = test_dir_path + "/matrix_transposed_inplace.npy";
    save_npy_matrix(mat, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/transpose_to_col.py";
    system(command);

    return 1;// TODO
}

int test_to_row_major(int rows, int columns) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    matrix<float> mat = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/matrix.npy";
    save_npy_matrix(mat, path);

    matrix<float> mat_transposed = to_row_major(&mat);
    path = test_dir_path + "/matrix_transposed.npy";
    save_npy_matrix(mat_transposed, path);

    to_row_major_inplace(&mat);
    path = test_dir_path + "/matrix_transposed_inplace.npy";
    save_npy_matrix(mat, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/transpose_to_row.py";
    system(command);

    return 1;// TODO
}


TEST_CASE("Transpose", "[transpose]") {
    CHECK(test_to_column_major(512, 1024));
    CHECK(test_to_column_major(4096, 256));
    CHECK(test_to_column_major(2003, 661));
    CHECK(test_to_column_major(2017, 389));
    CHECK(test_to_column_major(7919, 4007));

    CHECK(test_to_row_major(512, 1024));
    CHECK(test_to_row_major(4096, 256));
    CHECK(test_to_row_major(2003, 661));
    CHECK(test_to_row_major(2017, 389));
    CHECK(test_to_row_major(7919, 4007));
}
