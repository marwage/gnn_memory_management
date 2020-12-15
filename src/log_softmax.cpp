// Copyright 2020 Marcel Wagenländer

#include "log_softmax.hpp"


LogSoftmax::LogSoftmax() {}

LogSoftmax::LogSoftmax(CudaHelper *helper) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;
}

LogSoftmax::LogSoftmax(CudaHelper *helper, long num_nodes, long num_features) {
    set(helper, num_nodes, num_features);
}

void LogSoftmax::set(CudaHelper *helper, long num_nodes, long num_features) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

void LogSoftmax::forward(Matrix<float> *x, Matrix<float> *y) {
    if (y->num_rows_ != x->num_rows_ || y->num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }
    to_row_major_inplace(x);

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y->values_, y->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y->num_rows_, 1, 1, y->num_columns_));

    check_cudnn(cudnnSoftmaxForward(cuda_helper_->cudnn_handle,
                                    CUDNN_SOFTMAX_LOG,
                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha_, x_desc, d_x,
                                    &beta_, y_desc, d_y));

    check_cuda(cudaMemcpy(y->values_, d_y,
                          y->num_rows_ * y->num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y->is_row_major_ = true;

    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));
}

Matrix<float> *LogSoftmax::forward(Matrix<float> *x) {
    forward(x, &y_);

    return &y_;
}

void LogSoftmax::backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients) {
    to_row_major_inplace(incoming_gradients);
    to_row_major_inplace(y);

    cudnnTensorDescriptor_t y_desc;
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y->values_, y->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y->num_rows_, 1, 1, y->num_columns_));

    cudnnTensorDescriptor_t dy_desc;
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->num_rows_ * incoming_gradients->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_,
                          incoming_gradients->num_rows_ * incoming_gradients->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));

    cudnnTensorDescriptor_t dx_desc;
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, y->size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y->num_rows_, 1, 1, y->num_columns_));

    check_cudnn(cudnnSoftmaxBackward(cuda_helper_->cudnn_handle,
                                     CUDNN_SOFTMAX_LOG,
                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &alpha_, y_desc, d_y,
                                     dy_desc, d_dy,
                                     &beta_, dx_desc, d_dx));

    check_cuda(cudaMemcpy(gradients->values_, d_dx, gradients->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients->is_row_major_ = true;

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));
}

Matrix<float> *LogSoftmax::backward(Matrix<float> *incoming_gradients) {
    backward(incoming_gradients, &y_, &gradients_);

    return &gradients_;
}

LogSoftmaxChunked::LogSoftmaxChunked() {}

LogSoftmaxChunked::LogSoftmaxChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features);
}

void LogSoftmaxChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    chunk_size_ = chunk_size;
    cuda_helper_ = helper;
    log_softmax_layer_ = LogSoftmax(helper);
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
        y_.at(i).set(current_chunk_size, num_features, true);
        gradients_.at(i).set(current_chunk_size, num_features, true);
    }
}

std::vector<Matrix<float>> *LogSoftmaxChunked::forward(std::vector<Matrix<float>> *x) {
    for (int i = 0; i < x->size(); ++i) {
        to_row_major_inplace(&x->at(i));
    }

    for (int i = 0; i < num_chunks_; ++i) {
        log_softmax_layer_.forward(&x->at(i), &y_.at(i));
    }

    return &y_;
}

std::vector<Matrix<float>> *LogSoftmaxChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    for (int i = 0; i < incoming_gradients->size(); ++i) {
        to_row_major_inplace(&incoming_gradients->at(i));
    }

    for (int i = 0; i < num_chunks_; ++i) {
        log_softmax_layer_.backward(&incoming_gradients->at(i), &y_.at(i), &gradients_.at(i));
    }

    return &gradients_;
}
