// 2020 Marcel Wagenländer

#include "add.hpp"
#include "dense_computation.hpp"


Add::Add(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    cuda_helper_ = cuda_helper;
    y_.set(num_nodes, num_features, true);
}

Matrix<float> *Add::op(Matrix<float> *a, Matrix<float> *b) {
    mat_mat_add(cuda_helper_, a, b, &y_);

    return &y_;
}

AddChunked::AddChunked(CudaHelper *cuda_helper, long num_nodes, long chunk_size) {
    cuda_helper_ = cuda_helper;
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    y_ = std::vector<Matrix<float>>(num_chunks);
}

std::vector<Matrix<float>> *AddChunked::op(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b) {
    long num_chunks = a->size();
    for (int i = 0; i < num_chunks; ++i) {
        if (a->at(i).is_row_major_ != b->at(i).is_row_major_) {
            to_row_major_inplace(&a->at(i));
            to_row_major_inplace(&b->at(i));
        }
        y_.at(i).set(a->at(i).num_rows_, a->at(i).num_columns_, a->at(i).is_row_major_);
        mat_mat_add(cuda_helper_, &a->at(i), &b->at(i), &y_.at(i));
    }

    return &y_;
}
