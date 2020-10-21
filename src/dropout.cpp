// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cmath>


Dropout::Dropout(CudaHelper *helper) {
    cuda_helper_ = helper;
}

matrix<float> Dropout::forward(matrix<float> X) {
    float probability = 0.2f;
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));
    unsigned long long seed = rand();
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc_));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc_,
                                          cuda_helper_->cudnn_handle, probability,
                                          d_states, state_size_, seed));

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, X.rows, X.columns));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, Y.rows, Y.columns));

    void *d_X, *d_Y;
    check_cuda(cudaMalloc(&d_X, X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values, X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMalloc(&d_Y, Y.rows * Y.columns * sizeof(float)));

    void *d_reserve_space;
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size_));
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_desc_, x_descr, d_X,
                                    y_descr, d_Y,
                                    d_reserve_space, reserve_space_size_));

    Y.values = (float *) malloc(Y.rows * Y.columns * sizeof(float));
    check_cuda(cudaMemcpy(Y.values, d_Y, Y.rows * Y.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    reserve_space_ = reinterpret_cast<void *>(malloc(reserve_space_size_));
    check_cuda(cudaMemcpy(reserve_space_, d_reserve_space,
                          reserve_space_size_,
                          cudaMemcpyDeviceToHost));

    states_ = reinterpret_cast<void *>(malloc(state_size_));
    check_cuda(cudaMemcpy(states_, d_states, state_size_,
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_reserve_space));
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_Y));

    return Y;
}

matrix<float> Dropout::forward_chunked(matrix<float> X, int chunk_size) {
    int num_chunks = ceil((float) X.rows / (float) chunk_size);

    float probability = 0.2f;
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));
    unsigned long long seed = rand();
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc_));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc_,
                                          cuda_helper_->cudnn_handle, probability,
                                          d_states, state_size_, seed));

    to_row_major(&X);  // TODO Can we avoid this?

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;
    Y.values = (float *) malloc(Y.rows * Y.columns * sizeof(float));

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    void *d_X;
    void *d_Y;
    void *d_reserve_space;
    check_cuda(cudaMalloc(&d_X, chunk_size * X.columns * sizeof(float)));
    check_cuda(cudaMalloc(&d_Y, chunk_size * Y.columns * sizeof(float)));

    int last_chunk_size;
    if (num_chunks * chunk_size > X.rows) {
        last_chunk_size = X.rows - (num_chunks - 1) * chunk_size;
    } else {
        last_chunk_size = chunk_size;
    }

    void *reserve_spaces[num_chunks]; 

    int current_chunk_size = chunk_size;
    for (int c = 0; c < num_chunks; ++c) {
        if (c == (num_chunks - 1)) {
            current_chunk_size = last_chunk_size;
        }

        check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, 1, current_chunk_size, X.columns));
        check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
                                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, 1, current_chunk_size, Y.columns));

        check_cuda(cudaMemcpy(d_X, &X.values[c * chunk_size], current_chunk_size * X.columns * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (c == 0) {
            check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size_));
            check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));

        }
        check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                        dropout_desc_, x_descr, d_X,
                                        y_descr, d_Y,
                                        d_reserve_space, reserve_space_size_));


        check_cuda(cudaMemcpy(&Y.values[c * chunk_size], d_Y, current_chunk_size * Y.columns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        reserve_spaces[c] = reinterpret_cast<void *>(malloc(reserve_space_size_));
        check_cuda(cudaMemcpy(reserve_spaces[c], d_reserve_space,
                              reserve_space_size_,
                              cudaMemcpyDeviceToHost));
    }

    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_Y));
    check_cuda(cudaFree(d_reserve_space));

    to_column_major(&Y);  // TODO Can we avoid this?

    return Y;
}

matrix<float> Dropout::backward(matrix<float> in_gradients) {
    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, in_gradients.rows, in_gradients.columns));
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, in_gradients.rows * in_gradients.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, in_gradients.values,
                          in_gradients.rows * in_gradients.columns,
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, in_gradients.rows, in_gradients.columns));
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, in_gradients.rows * in_gradients.columns * sizeof(float)));

    void *d_reserve_space;
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cuda(cudaMemcpy(d_reserve_space, reserve_space_,
                          reserve_space_size_,
                          cudaMemcpyHostToDevice));

    // It is expected that reserveSpace was populated during a call to cudnnDropoutForward and has not been changed
    check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle,
                                     dropout_desc_,
                                     dy_desc, d_dy,
                                     dx_desc, d_dx,
                                     d_reserve_space, reserve_space_size_));

    matrix<float> grad_input;
    grad_input.rows = in_gradients.rows;
    grad_input.columns = in_gradients.columns;
    grad_input.values = reinterpret_cast<float *>(malloc(grad_input.rows * grad_input.columns * sizeof(float)));
    check_cuda(cudaMemcpy(grad_input.values, d_dx,
                          grad_input.rows * grad_input.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    //clean-up
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_reserve_space));

    return grad_input;
}
