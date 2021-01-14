// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <catch2/catch.hpp>


void forward_pipeline(std::vector<Matrix<float>> *x, std::vector<Matrix<float>> *y) {
    CudaHelper cuda_helper;
    long num_steps = 3;

    for (long i = 0; i < (long) x->size(); ++i) {
        to_row_major_inplace(&x->at(i));
    }

    std::vector<float *> d_x(num_steps);
    std::vector<float *> d_y(num_steps);
    for (long j = 0; j < num_steps; ++j) {
        check_cuda(cudaMalloc(&d_x.at(j), x->at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_y.at(j), y->at(0).size_ * sizeof(float)));
    }

    cudaStream_t stream_in;
    check_cuda(cudaStreamCreate(&stream_in));
    cudaStream_t stream_out;
    check_cuda(cudaStreamCreate(&stream_out));
    cudaStream_t stream_compute;
    check_cuda(cudaStreamCreate(&stream_compute));
    check_cublas(cublasSetStream(cuda_helper.cublas_handle, stream_compute));

    long num_chunks = x->size();
    float alpha = 1.0;
    long chunk_zero = 0;
    long chunk_one = 1;
    long chunk_two = 2;
    for (int i = 0; i < num_chunks + 2; ++i) {
        // update chunk offsets
        chunk_zero = (i / 3) * 3;         // every three steps jump by 3
        chunk_one = ((i - 1) / 3) * 3 + 1;// one tick behind and one number higher
        chunk_two = ((i - 2) / 3) * 3 + 2;// two ticks behind and two number higher

        if (i % 3 == 0) {
            // zero in, one out, two compute
            // zero in
            if (chunk_zero < num_chunks) {
                check_cuda(cudaMemcpyAsync(d_x.at(0), x->at(chunk_zero).values_, x->at(chunk_zero).size_ * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_in));
                check_cuda(cudaMemcpyAsync(d_y.at(0), y->at(chunk_zero).values_, y->at(chunk_zero).size_ * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_in));
            }

            // one out
            if (chunk_one < num_chunks && i > 2) {
                check_cuda(cudaMemcpyAsync(y->at(chunk_one).values_, d_y.at(1), y->at(chunk_one).size_ * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream_out));
                y->at(chunk_one).is_row_major_ = true;
            }

            // two computation
            if (chunk_two < num_chunks && i > 2) {
                check_cublas(cublasSaxpy(cuda_helper.cublas_handle, x->at(chunk_two).size_,
                                         &alpha, d_x.at(2), 1, d_y.at(2), 1));
            }
        } else if (i % 3 == 1) {
            // one in, two out, zero compute
            // one in
            if (chunk_one < num_chunks && i > 0) {
                check_cuda(cudaMemcpyAsync(d_x.at(1), x->at(chunk_one).values_, x->at(chunk_one).size_ * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_in));
                check_cuda(cudaMemcpyAsync(d_y.at(1), y->at(chunk_one).values_, y->at(chunk_one).size_ * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_in));
            }

            // two out
            if (chunk_two < num_chunks && i > 3) {
                check_cuda(cudaMemcpyAsync(y->at(chunk_two).values_, d_y.at(2), y->at(chunk_two).size_ * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream_out));
                y->at(chunk_one).is_row_major_ = true;
            }

            // zero compute
            if (chunk_zero < num_chunks && i > 0) {
                check_cublas(cublasSaxpy(cuda_helper.cublas_handle, x->at(chunk_zero).size_,
                                         &alpha, d_x.at(0), 1, d_y.at(0), 1));
            }
        } else if (i % 3 == 2) {
            // zero out,  two in, one compute
            // zero out
            if (chunk_zero < num_chunks && i > 1) {
                check_cuda(cudaMemcpyAsync(y->at(chunk_zero).values_, d_y.at(0),
                                           y->at(chunk_zero).size_ * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream_out));
                y->at(chunk_zero).is_row_major_ = true;
            }

            // two in
            if (chunk_two < num_chunks && i > 0) {
                check_cuda(cudaMemcpyAsync(d_x.at(2), x->at(chunk_two).values_, x->at(chunk_two).size_ * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_in));
                check_cuda(cudaMemcpyAsync(d_y.at(2), y->at(chunk_two).values_, y->at(chunk_two).size_ * sizeof(float),
                                           cudaMemcpyHostToDevice, stream_in));
            }

            // one compute
            if (chunk_one < num_chunks && i > 0) {
                check_cublas(cublasSaxpy(cuda_helper.cublas_handle, x->at(chunk_one).size_,
                                         &alpha, d_x.at(1), 1, d_y.at(1), 1));
            }
        }

        // sync all spanned calls
        check_cuda(cudaDeviceSynchronize());
    }

    cudaStreamDestroy(stream_in);
    cudaStreamDestroy(stream_out);
    cudaStreamDestroy(stream_compute);

    // free GPU memory
    for (int j = 0; j < num_steps; ++j) {
        check_cuda(cudaFree(d_x.at(j)));
        check_cuda(cudaFree(d_y.at(j)));
    }
}

TEST_CASE("Forward pipeline", "[forward][pipeline]") {
    long chunk_size = 3;
    long num_chunks = 8;
    long num_features = 5;
    std::vector<Matrix<float>> x(num_chunks);
    std::vector<Matrix<float>> y(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        if (i == num_chunks - 1) {
            x.at(i).set(chunk_size - 1, num_features, true);
            y.at(i).set(chunk_size - 1, num_features, true);
        } else {
            x.at(i).set(chunk_size, num_features, true);
            y.at(i).set(chunk_size, num_features, true);
        }

        x.at(i).set_values(i + 1);
        y.at(i).set_values(i + num_chunks + 1);
    }

    forward_pipeline(&x, &y);

    for (int i = 0; i < num_chunks; ++i) {
        print_matrix(&y.at(i));
    }
}