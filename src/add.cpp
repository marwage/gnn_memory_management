// Copyright 2020 Marcel Wagenländer

#include "add.hpp"
#include "dense_computation.hpp"

Add::Add() {}

Add::Add(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    Add::set(cuda_helper, num_nodes, num_features);
}

void Add::set(CudaHelper *cuda_helper, long num_nodes, long num_features) {
    name_ = "add";
    cuda_helper_ = cuda_helper;
    y_.set(num_nodes, num_features, true);
    gradients_ = std::vector<Matrix<float> *>(2);
}

std::string Add::get_name() {
    return name_;
}

Matrix<float> *Add::forward(Matrix<float> *a, Matrix<float> *b) {
    mat_mat_add(cuda_helper_, a, b, &y_);
    return &y_;
}

std::vector<Matrix<float> *> *Add::backward(Matrix<float> *incoming_gradients) {
    gradients_.at(0) = incoming_gradients;
    gradients_.at(1) = incoming_gradients;
    return &gradients_;
}

// CHUNKED --- CHUNKED --- CHUNKED

AddChunked::AddChunked() {}

AddChunked::AddChunked(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features) {
    AddChunked::set(cuda_helper, chunk_size, num_nodes, num_features);
}

AddChunked::AddChunked(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    AddChunked::set(cuda_helper, chunk_size, num_nodes, num_features, keep_allocation);
}

AddChunked::~AddChunked() {
    if (keep_allocation_) {
        AddChunked::free_gpu_memory();
    }
}

void AddChunked::set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features) {
    AddChunked::set(cuda_helper, chunk_size, num_nodes, num_features, false);
}

void AddChunked::set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    AddChunked::set_common(cuda_helper, chunk_size, num_nodes, num_features, keep_allocation);

    if (keep_allocation_) {
        AddChunked::allocate_gpu_memory();
    }
}

void AddChunked::set_common(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    name_ = "add_chunked";
    cuda_helper_ = cuda_helper;
    chunk_size_ = chunk_size;
    keep_allocation_ = keep_allocation;
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

    d_a_ = NULL;
    d_b_ = NULL;
}

void AddChunked::allocate_gpu_memory() {
    check_cuda(cudaMalloc(&d_a_, y_.at(0).size_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_b_, y_.at(0).size_ * sizeof(float)));
}

void AddChunked::free_gpu_memory() {
    check_cuda(cudaFree(d_a_));
    check_cuda(cudaFree(d_b_));
}

std::vector<Matrix<float>> *AddChunked::forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b) {
    if (a->size() != b->size()) {
        throw "Inputs have unequal number of chunks";
    }

    if (a->at(0).is_row_major_ != b->at(0).is_row_major_) {
        for (long i = 0; i < num_chunks_; ++i) {
            to_column_major_inplace(&a->at(i));
            to_column_major_inplace(&b->at(i));
        }
    }

    if (!keep_allocation_) {
        allocate_gpu_memory();
    }

    for (long i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_a_, a->at(i).values_, a->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_b_, b->at(i).values_, b->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        mat_mat_add_cuda(cuda_helper_, d_a_, d_b_, a->at(i).size_);

        // out
        check_cuda(cudaMemcpy(y_.at(i).values_, d_b_, y_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    if (!keep_allocation_) {
        free_gpu_memory();
    }

    return &y_;
}

AddGradientsChunked *AddChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    gradients_.a = incoming_gradients;
    gradients_.b = incoming_gradients;

    return &gradients_;
}

// PIPELINED --- PIPELINED --- PIPELINED

AddPipelined::AddPipelined() {}

AddPipelined::AddPipelined(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features) {
    AddPipelined::set(cuda_helper, chunk_size, num_nodes, num_features);
}

AddPipelined::AddPipelined(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    AddPipelined::set(cuda_helper, chunk_size, num_nodes, num_features, keep_allocation);
}

AddPipelined::~AddPipelined() {
    if (keep_allocation_) {
        AddPipelined::free_gpu_memory();
    }
}

void AddPipelined::set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features) {
    AddPipelined::set(cuda_helper, chunk_size, num_nodes, num_features, false);
}

void AddPipelined::set(CudaHelper *cuda_helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    AddChunked::set_common(cuda_helper, chunk_size, num_nodes, num_features, keep_allocation);

    name_ = "add_pipelining";
    keep_allocation_ = keep_allocation;
    num_steps_ = 2;
    d_a_ = std::vector<float *>(num_steps_);
    d_b_ = std::vector<float *>(num_steps_);
    d_c_ = std::vector<float *>(num_steps_);

    if (keep_allocation_) {
        AddPipelined::allocate_gpu_memory();
    }
}

void AddPipelined::forward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_a_.at(buffer), a_->at(chunk).values_, a_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cuda(cudaMemcpyAsync(d_b_.at(buffer), b_->at(chunk).values_, b_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
}

void AddPipelined::forward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(y_.at(chunk).values_, d_c_.at(buffer), y_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
}

void AddPipelined::forward_compute(long chunk, long buffer) {
    check_cuda(cudaMemcpy(d_c_.at(buffer), d_b_.at(buffer), y_.at(chunk).size_ * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    mat_mat_add_cuda(cuda_helper_, d_a_.at(buffer), d_c_.at(buffer), a_->at(chunk).size_);
}

void AddPipelined::allocate_gpu_memory() {
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_a_.at(i), y_.at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_b_.at(i), y_.at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_c_.at(i), y_.at(0).size_ * sizeof(float)));
    }
}

void AddPipelined::free_gpu_memory() {
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_a_.at(i)));
        check_cuda(cudaFree(d_b_.at(i)));
        check_cuda(cudaFree(d_c_.at(i)));
    }
}

std::vector<Matrix<float>> *AddPipelined::forward(std::vector<Matrix<float>> *a, std::vector<Matrix<float>> *b) {
    if (a->size() != b->size()) {
        throw "Inputs have unequal number of chunks";
    }
    if (a->at(0).is_row_major_ != b->at(0).is_row_major_) {
        for (long i = 0; i < num_chunks_; ++i) {
            to_column_major_inplace(&a->at(i));
            to_column_major_inplace(&b->at(i));
        }
    }
    a_ = a;
    b_ = b;

    if (!keep_allocation_) {
        AddPipelined::allocate_gpu_memory();
    }

    pipeline(true, num_chunks_);

    if (!keep_allocation_) {
        AddPipelined::free_gpu_memory();
    }

    return &y_;
}

void AddPipelined::backward_in(long chunk, long buffer) {
    throw "Function not used";
}

void AddPipelined::backward_out(long chunk, long buffer) {
    throw "Function not used";
}

void AddPipelined::backward_compute(long chunk, long buffer) {
    throw "Function not used";
}

AddGradientsChunked *AddPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {
    return AddChunked::backward(incoming_gradients);
}
