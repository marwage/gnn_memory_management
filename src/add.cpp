// 2020 Marcel Wagenländer

#include "add.hpp"
#include "dense_computation.hpp"


Add::Add(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    cuda_helper_ = cuda_helper;
    y_.set(num_nodes, num_features, true);
}

Matrix<float> *Add::forward(Matrix<float> *a, Matrix<float> *b) {
    mat_mat_add(cuda_helper_, a, b, &y_);

    return &y_;
}

AddGradients *Add::backward(Matrix<float> *incoming_gradients) {
    gradients_.a = incoming_gradients;
    gradients_.b = incoming_gradients;

    return &gradients_;
}

// CHUNKED --- CHUNKED --- CHUNKED

AddChunked::AddChunked() {}

AddChunked::AddChunked(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features) {
    set(cuda_helper, chunk_size, num_nodes, num_features);
}

void AddChunked::set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features) {
    cuda_helper_ = cuda_helper;
    chunk_size_ = chunk_size;
    num_chunks_ = ceil((float) num_nodes / (float) chunk_size);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    y_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (long i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }
        y_.at(i).set(current_chunk_size, num_features, false);
    }
}

std::vector<Matrix<float>> *AddChunked::forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b) {
    if (a->size() != b->size()) {
        throw "Inputs have unequal number of chunks";
    }

    if (a->at(0).is_row_major_ != b->at(0).is_row_major_) {
        for (long i = 0; i < num_chunks_; ++i) {
            to_row_major_inplace(&a->at(i));
            to_row_major_inplace(&b->at(i));
        }
    }

    float *d_a;
    check_cuda(cudaMalloc(&d_a, a->at(0).size_ * sizeof(float)));
    float *d_b;
    check_cuda(cudaMalloc(&d_b, b->at(0).size_ * sizeof(float)));

    for (long i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_a, a->at(i).values_, a->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_b, b->at(i).values_, b->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        mat_mat_add_cuda(cuda_helper_, d_a, d_b, a->at(i).size_);

        // out
        check_cuda(cudaMemcpy(y_.at(i).values_, d_b, y_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    check_cuda(cudaFree(d_a));
    check_cuda(cudaFree(d_b));

    return &y_;
}

AddGradientsChunked *AddChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    gradients_.a = incoming_gradients;
    gradients_.b = incoming_gradients;

    return &gradients_;
}

// PIPELINED --- PIPELINED --- PIPELINED

AddPipelined::AddPipelined(CudaHelper *cudaHelper, long chunkSize, long numNodes, long numFeatures)
    : AddChunked(cudaHelper, chunkSize, numNodes, numFeatures) {

    num_steps_ = 3;
    d_a_ = std::vector<float *>(num_steps_);
    d_b_ = std::vector<float *>(num_steps_);
}

void AddPipelined::forward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_a_.at(buffer), a_->at(chunk).values_, a_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cuda(cudaMemcpyAsync(d_b_.at(buffer), b_->at(chunk).values_, b_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
}

void AddPipelined::forward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(y_.at(chunk).values_, d_b_.at(buffer), y_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
}

void AddPipelined::forward_compute(long chunk, long buffer) {
    mat_mat_add_cuda(cuda_helper_, d_a_.at(buffer), d_b_.at(buffer), a_->at(chunk).size_);
}

std::vector<Matrix<float>> *AddPipelined::forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b) {
    a_ = a;
    b_ = b;

    if (a->size() != b->size()) {
        throw "Inputs have unequal number of chunks";
    }

    if (a->at(0).is_row_major_ != b->at(0).is_row_major_) {
        for (long i = 0; i < num_chunks_; ++i) {
            to_row_major_inplace(&a->at(i));
            to_row_major_inplace(&b->at(i));
        }
    }

    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_a_.at(i), a->at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_b_.at(i), b->at(0).size_ * sizeof(float)));
    }

    pipeline(true, num_chunks_);

    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_a_.at(i)));
        check_cuda(cudaFree(d_b_.at(i)));
    }

    return &y_;
}

AddGradientsChunked *AddPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {
    return AddChunked::backward(incoming_gradients);
}
