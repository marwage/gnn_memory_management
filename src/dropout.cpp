// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>


Dropout::Dropout() {}

Dropout::Dropout(CudaHelper *helper, long num_nodes, long num_features) {
    set(helper, num_nodes, num_features);
}

Dropout::~Dropout() {
    check_cuda(cudaFreeHost(reserve_space_));
}

void Dropout::set(CudaHelper *helper, long num_nodes, long num_features) {
    name_ = "dropout";
    cuda_helper_ = helper;
    probability_ = 0.2;
    seed_ = rand();

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);

    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
}

Matrix<float> *Dropout::forward(Matrix<float> *x) {
    is_row_major_ = x->is_row_major_;

    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));

    cudnnDropoutDescriptor_t dropout_desc;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states, state_size_, seed_));

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    void *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    void *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));

    void *d_reserve_space;
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size_));
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));

    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_desc, x_descr, d_x,
                                    y_descr, d_y,
                                    d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = is_row_major_;

    if (reserve_space_ == NULL) {
        check_cuda(cudaMallocHost(&reserve_space_, reserve_space_size_));
    }
    check_cuda(cudaMemcpy(reserve_space_, d_reserve_space,
                          reserve_space_size_,
                          cudaMemcpyDeviceToHost));

    // free
    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_reserve_space));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *Dropout::backward(Matrix<float> *incoming_gradients) {
    if (y_.num_rows_ != incoming_gradients->num_rows_ || y_.num_columns_ != incoming_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }
    if (is_row_major_ != incoming_gradients->is_row_major_) {
        to_major_inplace(incoming_gradients, is_row_major_);
    }

    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));

    cudnnDropoutDescriptor_t dropout_desc;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states, state_size_, seed_));

    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_,
                          incoming_gradients->size_,
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, incoming_gradients->size_ * sizeof(float)));

    void *d_reserve_space;
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cuda(cudaMemcpy(d_reserve_space, reserve_space_,
                          reserve_space_size_,
                          cudaMemcpyHostToDevice));

    // It is expected that reserveSpace was populated during a call to cudnnDropoutForward and has not been changed
    check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle,
                                     dropout_desc,
                                     dy_desc, d_dy,
                                     dx_desc, d_dx,
                                     d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(gradients_.values_, d_dx,
                          gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    gradients_.is_row_major_ = is_row_major_;

    // free
    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_reserve_space));

    return &gradients_;
}

// CHUNKED --- CHUNKED -- CHUNKED

DropoutChunked::DropoutChunked() {}

DropoutChunked::DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features);
}

DropoutChunked::DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features, bool keep_allocation) {
    set(helper, chunk_size, num_nodes, num_features, keep_allocation);
}

DropoutChunked::~DropoutChunked() {
    for (long i = 0; i < num_chunks_; ++i) {
        check_cuda(cudaFreeHost(reserve_space_.at(i)));
    }
    if (keep_allocation_) {
        free_gpu_memory();
    }
}

void DropoutChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features, false);
}

void DropoutChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    name_ = "dropout_chunked";
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    probability_ = 0.2;
    seed_ = rand();
    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    keep_allocation_ = keep_allocation;

    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    reserve_space_ = std::vector<char *>(num_chunks_);
    y_ = std::vector<Matrix<float>>(num_chunks_);
    gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }
        y_.at(i).set(current_chunk_size, num_features, true);
        gradients_.at(i).set(current_chunk_size, num_features, true);
    }

    if (keep_allocation_) {
        allocate_gpu_memory();
    }
}

void DropoutChunked::allocate_gpu_memory() {
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
    check_cuda(cudaMalloc(&d_states_, state_size_));

    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc_));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc_,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states_, state_size_, seed_));

    check_cuda(cudaMalloc(&d_x_, y_.at(0).size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc_));

    check_cuda(cudaMalloc(&d_y_, y_.at(0).size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc_));

    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.at(0).num_rows_, 1, 1, y_.at(0).num_columns_));
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_desc_, &reserve_space_size_));
    check_cuda(cudaMalloc(&d_reserve_space_, reserve_space_size_));
}

void DropoutChunked::free_gpu_memory() {
    check_cuda(cudaFree(d_states_));
    check_cudnn(cudnnDestroyDropoutDescriptor(dropout_desc_));
    check_cuda(cudaFree(d_x_));
    check_cudnn(cudnnDestroyTensorDescriptor(x_desc_));
    check_cuda(cudaFree(d_y_));
    check_cudnn(cudnnDestroyTensorDescriptor(y_desc_));
    check_cuda(cudaFree(d_reserve_space_));
}

std::vector<Matrix<float>> *DropoutChunked::forward(std::vector<Matrix<float>> *x) {
    is_row_major_ = x->at(0).is_row_major_;

    if (!keep_allocation_) {
        allocate_gpu_memory();
    }

    for (int i = 0; i < num_chunks_; ++i) {
        check_cuda(cudaMemcpy(d_x_, x->at(i).values_, x->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc_,
                                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               x->at(i).num_rows_, 1, 1, x->at(i).num_columns_));
        check_cudnn(cudnnSetTensor4dDescriptor(y_desc_,
                                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               y_.at(i).num_rows_, 1, 1, y_.at(i).num_columns_));

        check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                        dropout_desc_, x_desc_, d_x_,
                                        y_desc_, d_y_,
                                        d_reserve_space_, reserve_space_size_));

        check_cuda(cudaMemcpy(y_.at(i).values_, d_y_, y_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        y_.at(i).is_row_major_ = is_row_major_;

        if (reserve_space_.at(i) == NULL) {
            check_cuda(cudaMallocHost(&reserve_space_.at(i), reserve_space_size_));
        }
        check_cuda(cudaMemcpy(reserve_space_.at(i), d_reserve_space_, reserve_space_size_, cudaMemcpyDeviceToHost));
    }

    // free
    if (!keep_allocation_) {
        free_gpu_memory();
    }

    return &y_;
}

std::vector<Matrix<float>> *DropoutChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    if (is_row_major_ != incoming_gradients->at(0).is_row_major_) {
        for (int i = 0; i < num_chunks_; ++i) {
            to_major_inplace(&incoming_gradients->at(i), is_row_major_);
        }
    }

    if (!keep_allocation_) {
        allocate_gpu_memory();
    }

    for (int i = 0; i < num_chunks_; ++i) {
        check_cuda(cudaMemcpy(d_y_, incoming_gradients->at(i).values_, incoming_gradients->at(i).size_, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_reserve_space_, reserve_space_.at(i), reserve_space_size_, cudaMemcpyHostToDevice));
        check_cudnn(cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               incoming_gradients->at(i).num_rows_, 1, 1, incoming_gradients->at(i).num_columns_));
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               incoming_gradients->at(i).num_rows_, 1, 1, incoming_gradients->at(i).num_columns_));

        check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle, dropout_desc_,
                                         y_desc_, d_y_, x_desc_, d_x_, d_reserve_space_, reserve_space_size_));

        check_cuda(cudaMemcpy(gradients_.at(i).values_, d_x_, gradients_.at(i).size_ * sizeof(float), cudaMemcpyDeviceToHost));
        gradients_.at(i).is_row_major_ = is_row_major_;
    }

    // free
    if (!keep_allocation_) {
        free_gpu_memory();
    }

    return &gradients_;
}

// PIPELINED -- PIPELINED -- PIPELINED

DropoutPipelined::DropoutPipelined() {}

DropoutPipelined::DropoutPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features);
}

void DropoutPipelined::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    DropoutChunked::set(helper, chunk_size, num_nodes, num_features);

    name_ = "dropout_pipelined";
    num_steps_ = 2;
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));

    d_states_ = std::vector<char *>(num_steps_);
    d_reserve_space_ = std::vector<char *>(num_steps_);
    d_x_ = std::vector<float *>(num_steps_);
    x_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    d_y_ = std::vector<float *>(num_steps_);
    y_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    d_dx_ = std::vector<float *>(num_steps_);
    dx_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    d_dy_ = std::vector<float *>(num_steps_);
    dy_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    dropout_desc_ = std::vector<cudnnDropoutDescriptor_t>(num_steps_);
}

void DropoutPipelined::forward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_x_.at(buffer), x_->at(chunk).values_, x_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x_->at(chunk).num_rows_, 1, 1, x_->at(chunk).num_columns_));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.at(chunk).num_rows_, 1, 1, y_.at(chunk).num_columns_));
}

void DropoutPipelined::forward_out(long chunk, long buffer) {
    if (reserve_space_.at(chunk) == NULL) {
        check_cuda(cudaMallocHost(&reserve_space_.at(chunk), reserve_space_size_));
    }
    check_cuda(cudaMemcpyAsync(reserve_space_.at(chunk), d_reserve_space_.at(buffer), reserve_space_size_,
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));

    check_cuda(cudaMemcpyAsync(y_.at(chunk).values_, d_y_.at(buffer), y_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
    y_.at(chunk).is_row_major_ = is_row_major_;
}

void DropoutPipelined::forward_compute(long chunk, long buffer) {
    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_desc_.at(buffer), x_desc_.at(buffer), d_x_.at(buffer),
                                    y_desc_.at(buffer), d_y_.at(buffer),
                                    d_reserve_space_.at(buffer), reserve_space_size_));
}

void DropoutPipelined::backward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_dy_.at(buffer), incoming_gradients_->at(chunk).values_, incoming_gradients_->at(chunk).size_,
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cuda(cudaMemcpyAsync(d_reserve_space_.at(buffer), reserve_space_.at(chunk), reserve_space_size_,
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients_->at(chunk).num_rows_, 1, 1, incoming_gradients_->at(chunk).num_columns_));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients_->at(chunk).num_rows_, 1, 1, incoming_gradients_->at(chunk).num_columns_));
}

void DropoutPipelined::backward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(gradients_.at(chunk).values_, d_dx_.at(buffer), gradients_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
    gradients_.at(chunk).is_row_major_ = is_row_major_;
}

void DropoutPipelined::backward_compute(long chunk, long buffer) {
    check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle, dropout_desc_.at(buffer),
                                     dy_desc_.at(buffer), d_dy_.at(buffer), dx_desc_.at(buffer), d_dx_.at(buffer),
                                     d_reserve_space_.at(buffer), reserve_space_size_));
}

std::vector<Matrix<float>> *DropoutPipelined::forward(std::vector<Matrix<float>> *x) {
    x_ = x;
    is_row_major_ = x->at(0).is_row_major_;

    // allocate
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_states_.at(i), state_size_));

        check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc_.at(i)));
        check_cudnn(cudnnSetDropoutDescriptor(dropout_desc_.at(i), cuda_helper_->cudnn_handle, probability_,
                                              d_states_.at(i), state_size_, seed_));

        check_cuda(cudaMalloc(&d_x_.at(i), x->at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&x_desc_.at(i)));
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc_.at(i), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               x->at(0).num_rows_, 1, 1, x->at(0).num_columns_));

        check_cuda(cudaMalloc(&d_y_.at(i), y_.at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&y_desc_.at(i)));

        check_cudnn(cudnnDropoutGetReserveSpaceSize(x_desc_.at(i), &reserve_space_size_));
        check_cuda(cudaMalloc(&d_reserve_space_.at(i), reserve_space_size_));
    }

    pipeline(true, num_chunks_);

    // free
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_states_.at(i)));

        check_cudnn(cudnnDestroyDropoutDescriptor(dropout_desc_.at(i)));

        check_cuda(cudaFree(d_x_.at(i)));
        check_cudnn(cudnnDestroyTensorDescriptor(x_desc_.at(i)));

        check_cuda(cudaFree(d_y_.at(i)));
        check_cudnn(cudnnDestroyTensorDescriptor(y_desc_.at(i)));

        check_cuda(cudaFree(d_reserve_space_.at(i)));
    }

    return &y_;
}

std::vector<Matrix<float>> *DropoutPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {
    incoming_gradients_ = incoming_gradients;
    if (is_row_major_ != incoming_gradients->at(0).is_row_major_) {
        for (int i = 0; i < num_chunks_; ++i) {
            to_major_inplace(&incoming_gradients->at(i), is_row_major_);
        }
    }

    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_states_.at(i), state_size_));

        check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc_.at(i)));
        check_cudnn(cudnnSetDropoutDescriptor(dropout_desc_.at(i), cuda_helper_->cudnn_handle, probability_,
                                              d_states_.at(i), state_size_, seed_));

        check_cuda(cudaMalloc(&d_dy_.at(i), incoming_gradients->at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&dy_desc_.at(i)));

        check_cuda(cudaMalloc(&d_dx_.at(i), x_->at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&dx_desc_.at(i)));

        check_cuda(cudaMalloc(&d_reserve_space_.at(i), reserve_space_size_));
    }

    pipeline(false, num_chunks_);

    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_states_.at(i)));

        check_cudnn(cudnnDestroyDropoutDescriptor(dropout_desc_.at(i)));

        check_cuda(cudaFree(d_dy_.at(i)));
        check_cudnn(cudnnDestroyTensorDescriptor(dy_desc_.at(i)));

        check_cuda(cudaFree(d_dx_.at(i)));
        check_cudnn(cudnnDestroyTensorDescriptor(dx_desc_.at(i)));

        check_cuda(cudaFree(d_reserve_space_.at(i)));
    }

    return &gradients_;
}
