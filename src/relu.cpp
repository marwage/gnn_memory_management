// Copyright 2020 Marcel Wagenländer

#include "relu.hpp"

#include <cmath>
#include <limits>


Relu::Relu() {}

Relu::Relu(CudaHelper *helper, long num_nodes, long num_features) {
    set(helper, num_nodes, num_features);
}

void Relu::set(CudaHelper *helper, long num_nodes, long num_features) {
    name_ = "relu";
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;

    check_cudnn(cudnnCreateActivationDescriptor(&relu_desc_));
    double coef = std::numeric_limits<double>::max();
    check_cudnn(cudnnSetActivationDescriptor(relu_desc_,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coef));

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *Relu::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }
    x_ = x;

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    check_cudnn(cudnnActivationForward(cuda_helper_->cudnn_handle,
                                       relu_desc_,
                                       &alpha_, x_desc, d_x,
                                       &beta_, y_desc, d_y));

    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = true;

    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *Relu::backward(Matrix<float> *incoming_gradients) {
    to_row_major_inplace(incoming_gradients);
    to_row_major_inplace(&y_);
    to_row_major_inplace(x_);

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float), cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_, incoming_gradients->size_ * sizeof(float), cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x_->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x_->values_, x_->size_ * sizeof(float), cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x_->num_rows_, 1, 1, x_->num_columns_));

    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, x_->size_ * sizeof(float)));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x_->num_rows_, 1, 1, x_->num_columns_));

    check_cudnn(cudnnActivationBackward(cuda_helper_->cudnn_handle,
                                        relu_desc_,
                                        &alpha_, y_desc, d_y,
                                        dy_desc, d_dy,
                                        x_desc, d_x,
                                        &beta_, dx_desc, d_dx));

    check_cuda(cudaMemcpy(gradients_.values_, d_dx,
                          gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = true;

    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));

    return &gradients_;
}

// CHUNKED --- CHUNKED --- CHUNKED

ReluChunked::ReluChunked() {}

ReluChunked::ReluChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features);
}

ReluChunked::ReluChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    set(helper, chunk_size, num_nodes, num_features, keep_allocation);
}

ReluChunked::~ReluChunked() {
    if (keep_allocation_) {
        free_gpu_memory();
    }
}

void ReluChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features, false);
}

void ReluChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) {
    name_ = "relu_chunked";
    chunk_size_ = chunk_size;
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;
    keep_allocation_ = keep_allocation;

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    y_ = std::vector<Matrix<float>>(num_chunks_);
    gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }
        y_[i].set(current_chunk_size, num_features, true);
        gradients_[i].set(current_chunk_size, num_features, true);
    }

    check_cudnn(cudnnCreateActivationDescriptor(&relu_desc_));
    double coef = std::numeric_limits<double>::max();
    check_cudnn(cudnnSetActivationDescriptor(relu_desc_,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coef));

    if (keep_allocation_) {
        allocate_gpu_memory();
    }
}

void ReluChunked::allocate_gpu_memory_forward() {
    check_cuda(cudaMalloc(&d_x_, y_.at(0).size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc_));

    check_cuda(cudaMalloc(&d_y_, y_.at(0).size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc_));
}

void ReluChunked::allocate_gpu_memory_backward() {
    ReluChunked::allocate_gpu_memory();
}

void ReluChunked::allocate_gpu_memory() {
    ReluChunked::allocate_gpu_memory_forward();

    check_cuda(cudaMalloc(&d_dx_, y_.at(0).size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc_));

    check_cuda(cudaMalloc(&d_dy_, y_.at(0).size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc_));
}

void ReluChunked::free_gpu_memory_forward() {
    check_cuda(cudaFree(d_x_));
    check_cudnn(cudnnDestroyTensorDescriptor(x_desc_));

    check_cuda(cudaFree(d_y_));
    check_cudnn(cudnnDestroyTensorDescriptor(y_desc_));
}

void ReluChunked::free_gpu_memory_backward() {
    ReluChunked::free_gpu_memory();
}

void ReluChunked::free_gpu_memory() {
    ReluChunked::free_gpu_memory_forward();

    check_cuda(cudaFree(d_dx_));
    check_cudnn(cudnnDestroyTensorDescriptor(dx_desc_));

    check_cuda(cudaFree(d_dy_));
    check_cudnn(cudnnDestroyTensorDescriptor(dy_desc_));
}

std::vector<Matrix<float>> *ReluChunked::forward(std::vector<Matrix<float>> *x) {
    if (num_chunks_ != (long) x->size()) {
        throw "Input has wrong number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_row_major_inplace(&x->at(i));
    }

    if (!keep_allocation_) {
        allocate_gpu_memory_forward();
    }

    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_x_, x->at(i).values_, x->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc_,
                                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               x->at(i).num_rows_, 1, 1, x->at(i).num_columns_));
        check_cudnn(cudnnSetTensor4dDescriptor(y_desc_,
                                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               y_.at(i).num_rows_, 1, 1, y_.at(i).num_columns_));

        // compute
        check_cudnn(cudnnActivationForward(cuda_helper_->cudnn_handle,
                                           relu_desc_,
                                           &alpha_, x_desc_, d_x_,
                                           &beta_, y_desc_, d_y_));

        // out
        check_cuda(cudaMemcpy(y_.at(i).values_, d_y_,
                              y_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        y_.at(i).is_row_major_ = true;
    }

    // free GPU memory
    if (!keep_allocation_) {
        free_gpu_memory_forward();
    }

    x_ = x;

    return &y_;
}

std::vector<Matrix<float>> *ReluChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    if (num_chunks_ != (long) incoming_gradients->size()) {
        throw "Incoming gradients has wrong number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_row_major_inplace(&incoming_gradients->at(i));
        to_row_major_inplace((&x_->at(i)));
        to_row_major_inplace((&y_.at(i)));
    }

    if (!keep_allocation_) {
        ReluChunked::allocate_gpu_memory_backward();
    }

    for (int i = 0; i < num_chunks_; ++i) {
        check_cuda(cudaMemcpy(d_y_, y_.at(i).values_, y_.at(i).size_ * sizeof(float), cudaMemcpyHostToDevice));
        check_cudnn(cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               y_.at(i).num_rows_, 1, 1, y_.at(i).num_columns_));
        check_cuda(cudaMemcpy(d_dy_, incoming_gradients->at(i).values_, incoming_gradients->at(i).size_ * sizeof(float), cudaMemcpyHostToDevice));
        check_cudnn(cudnnSetTensor4dDescriptor(dy_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               incoming_gradients->at(i).num_rows_, 1, 1, incoming_gradients->at(i).num_columns_));
        check_cuda(cudaMemcpy(d_x_, x_->at(i).values_, x_->at(i).size_ * sizeof(float), cudaMemcpyHostToDevice));
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               x_->at(i).num_rows_, 1, 1, x_->at(i).num_columns_));
        check_cudnn(cudnnSetTensor4dDescriptor(dx_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               x_->at(i).num_rows_, 1, 1, x_->at(i).num_columns_));

        check_cudnn(cudnnActivationBackward(cuda_helper_->cudnn_handle,
                                            relu_desc_,
                                            &alpha_, y_desc_, d_y_,
                                            dy_desc_, d_dy_,
                                            x_desc_, d_x_,
                                            &beta_, dx_desc_, d_dx_));

        check_cuda(cudaMemcpy(gradients_.at(i).values_, d_dx_,
                              gradients_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        gradients_.at(i).is_row_major_ = true;
    }

    // free
    if (!keep_allocation_) {
        free_gpu_memory_backward();
    }

    return &gradients_;
}

// PIPELINED --- PIPELINED --- PIPELINED

ReluPipelined::ReluPipelined() {}

ReluPipelined::ReluPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features);
}

void ReluPipelined::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    ReluChunked::set(helper, chunk_size, num_nodes, num_features);

    name_ = "relu_pipelined";
    num_steps_ = 2;
    d_x_ = std::vector<float *>(num_steps_);
    x_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    d_y_ = std::vector<float *>(num_steps_);
    y_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    d_dx_ = std::vector<float *>(num_steps_);
    dx_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
    d_dy_ = std::vector<float *>(num_steps_);
    dy_desc_ = std::vector<cudnnTensorDescriptor_t>(num_steps_);
}

void ReluPipelined::forward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_x_.at(buffer), x_->at(chunk).values_, x_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x_->at(chunk).num_rows_, 1, 1, x_->at(chunk).num_columns_));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.at(chunk).num_rows_, 1, 1, y_.at(chunk).num_columns_));
}

void ReluPipelined::forward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(y_.at(chunk).values_, d_y_.at(buffer), y_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
    y_.at(chunk).is_row_major_ = true;
}

void ReluPipelined::forward_compute(long chunk, long buffer) {
    check_cudnn(cudnnActivationForward(cuda_helper_->cudnn_handle,
                                       relu_desc_,
                                       &alpha_, x_desc_.at(buffer), d_x_.at(buffer),
                                       &beta_, y_desc_.at(buffer), d_y_.at(buffer)));
}

void ReluPipelined::backward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_y_.at(buffer), y_.at(chunk).values_, y_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cuda(cudaMemcpyAsync(d_dy_.at(buffer), incoming_gradients_->at(chunk).values_, incoming_gradients_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cuda(cudaMemcpyAsync(d_x_.at(buffer), x_->at(chunk).values_, x_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    // no copy for d_dx needed

    check_cudnn(cudnnSetTensor4dDescriptor(y_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.at(chunk).num_rows_, 1, 1, y_.at(chunk).num_columns_));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.at(chunk).num_rows_, 1, 1, y_.at(chunk).num_columns_));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x_->at(chunk).num_rows_, 1, 1, x_->at(chunk).num_columns_));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc_.at(buffer), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x_->at(chunk).num_rows_, 1, 1, x_->at(chunk).num_columns_));
}

void ReluPipelined::backward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(gradients_.at(chunk).values_, d_dx_.at(buffer), gradients_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
    gradients_.at(chunk).is_row_major_ = true;
}

void ReluPipelined::backward_compute(long chunk, long buffer) {
    check_cudnn(cudnnActivationBackward(cuda_helper_->cudnn_handle,
                                        relu_desc_,
                                        &alpha_, y_desc_.at(buffer), d_y_.at(buffer),
                                        dy_desc_.at(buffer), d_dy_.at(buffer),
                                        x_desc_.at(buffer), d_x_.at(buffer),
                                        &beta_, dx_desc_.at(buffer), d_dx_.at(buffer)));
}

std::vector<Matrix<float>> *ReluPipelined::forward(std::vector<Matrix<float>> *x) {
    x_ = x;

    if ((long) x->size() != num_chunks_) {
        throw "Input has wrong number of chunks";
    }
    for (long i = 0; i < num_chunks_; ++i) {
        to_row_major_inplace(&x->at(i));
    }

    // allocate
    for (long j = 0; j < num_steps_; ++j) {
        check_cuda(cudaMalloc(&d_x_.at(j), x->at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&x_desc_.at(j)));

        check_cuda(cudaMalloc(&d_y_.at(j), y_.at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&y_desc_.at(j)));
    }

    pipeline(true, num_chunks_);

    // free
    for (long j = 0; j < num_steps_; ++j) {
        check_cuda(cudaFree(d_x_.at(j)));
        check_cudnn(cudnnDestroyTensorDescriptor(x_desc_.at(j)));

        check_cuda(cudaFree(d_y_.at(j)));
        check_cudnn(cudnnDestroyTensorDescriptor(y_desc_.at(j)));
    }

    return &y_;
}

std::vector<Matrix<float>> *ReluPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {
    incoming_gradients_ = incoming_gradients;

    if ((long) incoming_gradients->size() != num_chunks_) {
        throw "Incoming gradients has wrong number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_row_major_inplace(&incoming_gradients->at(i));
        to_row_major_inplace((&x_->at(i)));
        to_row_major_inplace((&y_.at(i)));
    }

    // allocate
    for (long j = 0; j < num_steps_; ++j) {
        check_cuda(cudaMalloc(&d_x_.at(j), x_->at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&x_desc_.at(j)));

        check_cuda(cudaMalloc(&d_y_.at(j), y_.at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&y_desc_.at(j)));

        check_cuda(cudaMalloc(&d_dx_.at(j), x_->at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&dx_desc_.at(j)));

        check_cuda(cudaMalloc(&d_dy_.at(j), y_.at(0).size_ * sizeof(float)));
        check_cudnn(cudnnCreateTensorDescriptor(&dy_desc_.at(j)));
    }

    pipeline(false, num_chunks_);

    // free
    for (long j = 0; j < num_steps_; ++j) {
        check_cuda(cudaFree(d_x_.at(j)));
        check_cudnn(cudnnDestroyTensorDescriptor(x_desc_.at(j)));

        check_cuda(cudaFree(d_y_.at(j)));
        check_cudnn(cudnnDestroyTensorDescriptor(y_desc_.at(j)));

        check_cuda(cudaFree(d_dx_.at(j)));
        check_cudnn(cudnnDestroyTensorDescriptor(dx_desc_.at(j)));

        check_cuda(cudaFree(d_dy_.at(j)));
        check_cudnn(cudnnDestroyTensorDescriptor(dy_desc_.at(j)));
    }

    return &gradients_;
}
