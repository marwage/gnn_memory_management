// Copyright 2020 Marcel Wagenländer

#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


int test_transpose(long rows, long columns, bool to_col_major) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    matrix<float> mat = gen_rand_matrix(rows, columns);
    if (!to_col_major) {
        mat.row_major = false;
    }
    path = test_dir_path + "/matrix.npy";
    save_npy_matrix_no_trans(mat, path);

    if (to_col_major) {
        to_column_major_inplace(&mat);
    } else {
        to_row_major_inplace(&mat);
    }
    path = test_dir_path + "/matrix_transposed.npy";
    save_npy_matrix_no_trans(mat, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/transpose.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("Transpose", "[transpose]") {
    CHECK(test_transpose(99999, 6584, true));
    CHECK(test_transpose(85647, 6584, true));
    CHECK(test_transpose(84634, 8573, true));
}
// CHECK(test_transpose(x, y, true)); python script would need to be aware of the bool
