#include <random>
#include <math.h>
#include <limits>

#include "cnpy.h"
#include "mmio_wrapper.h"

#include <cuda_runtime.h>
#include "cusparse.h"
#include <cudnn.h>
#include <cublas_v2.h>


void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("CUDA API failed at line %d with error: %s (%d)\n",
                __LINE__, cudaGetErrorString(status), status);
    }
}

void check_cusparse(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",
                __LINE__, cusparseGetErrorString(status), status);
    }
}

void check_cudnn(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("CUDNN API failed at line %d with error: %s (%d)\n",
                __LINE__, cudnnGetErrorString(status), status);
    }
}

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS API failed at line %d with error: %s (%d)\n",
                __LINE__, cublasGetErrorString(status), status);
    }
}

template <typename T>
void print_matrix(T* a, int rows, int cols) {
    int N;
    if (rows < 10) {
        N = rows;
    } else {
        N = 10;
    }
    int M;
    if (cols < 10) {
        M = cols;
    } else {
        M = 10;
    }

    // for (int i = 0; i < rows; i = i + 1) {
    for (int i = 0; i < N; i = i + 1) {
        // for (int j = 0; j < cols; j = j + 1) {
        for (int j = 0; j < M; j = j + 1) {
            std::cout << a[i * cols + j] << ",";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print_vector(T* a, int num_ele) {
    print_matrix(a, num_ele, 1);
}

int new_index(int old_idx, int N, int M) {
    int last_idx = M * N - 1;
    if (old_idx == last_idx) {
        return last_idx;
    } else {
        long int new_idx = (long int) N * (long int) old_idx;
        new_idx = new_idx % last_idx;
        return (int) new_idx;
    }
}

template <typename T>
void transpose(T *a_T, T *a, int rows, int cols) {
    int old_idx, new_idx;
    for (int i = 0; i < rows; i = i + 1) {
        for (int j = 0; j < cols; j = j + 1) {
            old_idx = i * cols + j;
            new_idx = new_index(old_idx, rows, cols);
            a_T[new_idx] = a[old_idx];
        }
    }
}

template <typename T>
void one_to_zero_index(T *a, int len) {
    for (int i = 0; i < len; ++i) {
        a[i] = a[i] - 1;
    }
}

template <typename T>
struct matrix {
    int rows;
    int columns;
    T *values;
};

template <typename T>
struct vector {
    int size;
    T *values;
};

template <typename T>
struct sparse_matrix {
    int rows;
    int columns;
    int nnz;
    T *csr_val;
    int *csr_row_ptr;
    int *csr_col_ind;
};

matrix<float> graph_convolution(sparse_matrix<float> A, matrix<float> B,
        std::string reduction) {
    bool mean;
    if (reduction.compare("mean")) {
        mean = true;
    } else if (reduction.compare("sum")) {
        mean = false;
    } else {
        std::cout << "Reduction not supported" << std::endl;
    }

    cudaError_t cuda_error = cudaSuccess;
    cusparseStatus_t sparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparseHandle_t sparse_handle;
    sparse_status = cusparseCreate(&sparse_handle);
    check_cusparse(sparse_status);


    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    cuda_error = cudaMalloc((void**) &d_A_csr_val, 
            A.nnz * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMalloc((void**) &d_A_csr_row_offsets,
            (A.rows + 1) * sizeof(int));
    check_cuda(cuda_error);
    cuda_error = cudaMalloc((void**) &d_A_col_ind,
            A.nnz * sizeof(int));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_A_csr_val, A.csr_val,
            A.nnz * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_A_csr_row_offsets, A.csr_row_ptr,
            (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
   cuda_error = cudaMemcpy(d_A_col_ind, A.csr_col_ind,
            A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cusparseSpMatDescr_t A_descr;
    sparse_status = cusparseCreateCsr(&A_descr, A.rows,
            A.columns, A.nnz,
            d_A_csr_row_offsets, d_A_col_ind,
            d_A_csr_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    check_cusparse(sparse_status);
  
    // features to column-major
    matrix<float> B_col;
    B_col.rows = B.columns;
    B_col.columns = B.rows;
    B_col.values = (float*) malloc(B_col.rows * B_col.columns * sizeof(float));
    transpose<float>(B_col.values, B.values, B.rows, B.columns);
    // create cusparse features
    float *d_B;
    cuda_error = cudaMalloc((void**) &d_B, B.rows * B.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_B, B_col.values, B.rows * B.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cusparseDnMatDescr_t B_descr;
    sparse_status = cusparseCreateDnMat(&B_descr, B.rows, B.columns,
            B.rows, d_B,
            CUDA_R_32F, CUSPARSE_ORDER_COL);
    check_cusparse(sparse_status);

    // create result
    matrix<float> result;
    result.rows = A.rows;
    result.columns = B.columns;
    result.values = (float*) malloc(result.rows * result.columns * sizeof(float));
    for (int i = 0; i < result.rows * result.columns; ++i) {
        result.values[i] = 0.0f;
    }
    // result to column-major
    matrix<float> result_col;
    result_col.rows = result.columns;
    result_col.columns = result.rows;
    result_col.values = (float*) malloc(result_col.rows * result_col.columns * sizeof(float));
    transpose<float>(result_col.values, result.values, result.rows, result.columns);

    // create cusparse result
    float *d_result;
    cuda_error = cudaMalloc((void**) &d_result, result.rows * result.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_result, result_col.values, result_col.rows * result_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cusparseDnMatDescr_t result_descr;
    sparse_status = cusparseCreateDnMat(&result_descr, result.rows, result.columns,
            result.rows, d_result,
            CUDA_R_32F, CUSPARSE_ORDER_COL);
    check_cusparse(sparse_status);

    // get buffer size for SpMM
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t buffer_size;
    sparse_status = cusparseSpMM_bufferSize(sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A_descr, B_descr, &beta, result_descr,
            // CUSPARSE_MM_ALG_DEFAULT is deprecated
            // but CUSPARSE_SPMM_ALG_DEFAULT is not working
            CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
            &buffer_size);
    check_cusparse(sparse_status);
    void *d_buffer;
    cuda_error = cudaMalloc(&d_buffer, buffer_size);
    check_cuda(cuda_error);

    // compute SpMM
    sparse_status = cusparseSpMM(sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A_descr, B_descr, &beta, result_descr,
            CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
            d_buffer);
    check_cusparse(sparse_status);

    cuda_error = cudaFree(d_buffer);
    check_cuda(cuda_error);

    // move result_col to CPU memory
    cuda_error = cudaMemcpy(result_col.values, d_result,
            result_col.rows * result_col.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

    // result to row-major
    transpose<float>(result.values, result_col.values, result_col.rows, result_col.columns);

    // apply mean
    if (mean) {
        vector<float> ones;
        ones.size = A.rows;
        ones.values = (float *) malloc(ones.size * sizeof(float));
        for (int i = 0; i < ones.size; ++i) {
            ones.values[i] = 1.0;
        }
        float *d_ones;
        cuda_error = cudaMalloc(&d_ones, ones.size * sizeof(float));
        check_cuda(cuda_error);
        cuda_error = cudaMemcpy(d_ones, ones.values, ones.size * sizeof(float),
                cudaMemcpyHostToDevice);
        check_cuda(cuda_error);
        cusparseDnVecDescr_t ones_desc;
        sparse_status = cusparseCreateDnVec(&ones_desc, ones.size,
                d_ones, CUDA_R_32F);
        check_cusparse(sparse_status);

        vector<float> sum;
        sum.size = ones.size;
        sum.values = (float *) malloc(sum.size * sizeof(float));
        for (int i = 0; i < sum.size; ++i) {
            sum.values[0] = 0.0;
        }
        float *d_sum;
        cuda_error = cudaMalloc(&d_sum, sum.size * sizeof(float));
        check_cuda(cuda_error);
        cuda_error = cudaMemcpy(d_sum, sum.values, sum.size * sizeof(float),
                cudaMemcpyHostToDevice);
        cusparseDnVecDescr_t sum_desc;
        sparse_status = cusparseCreateDnVec(&sum_desc, sum.size,
                d_sum, CUDA_R_32F);
        check_cusparse(sparse_status);

        sparse_status = cusparseSpMV_bufferSize(sparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, A_descr, ones_desc,
                &beta, sum_desc,
                CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer_size);
        check_cusparse(sparse_status);
        cuda_error = cudaMalloc(&d_buffer, buffer_size);
        check_cuda(cuda_error);
        sparse_status = cusparseSpMV(sparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, A_descr, ones_desc,
                &beta, sum_desc,
                CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, d_buffer);
        check_cusparse(sparse_status);

        cuda_error = cudaMemcpy(sum.values, d_sum,
                sum.size * sizeof(float),
                cudaMemcpyDeviceToHost);
        check_cuda(cuda_error);

        // TODO do the following on GPU
        // scale by 1 / sum
        for (int i = 0; i < result.rows; ++i) {
            for (int j = 0; j < result.columns; ++j) {
                result.values[i * result.columns + j] = result.values[i * result.columns + j] / sum.values[i];
            }
        }

        // free GPU memory
        cuda_error = cudaFree(d_ones);
        check_cuda(cuda_error);
        cuda_error = cudaFree(d_sum);
        check_cuda(cuda_error);

        // free CPU memory
        free(ones.values);
        free(sum.values);
    }  // end mean

    // free memory
    cuda_error = cudaFree(d_A_csr_val);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_A_col_ind);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_A_csr_row_offsets);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_B);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_buffer);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_result);
    check_cuda(cuda_error);

    sparse_status = cusparseDestroy(sparse_handle);
    check_cusparse(sparse_status);

    return result;

}

matrix<float> dropout(matrix<float> X) {
    cudaError_t cuda_error;
    cudnnStatus_t cudnn_status;
    cudnnHandle_t cudnn_handle;
    cudnn_status = cudnnCreate(&cudnn_handle);

    float probability = 0.2f;
    size_t state_size;
    cudnn_status = cudnnDropoutGetStatesSize(cudnn_handle, &state_size);
    void *states;
    cuda_error = cudaMalloc(&states, state_size);
    check_cuda(cuda_error);
    unsigned long long seed = rand();
    cudnnDropoutDescriptor_t dropout_descr;
    cudnn_status = cudnnCreateDropoutDescriptor(&dropout_descr);
    check_cudnn(cudnn_status);
    cudnn_status = cudnnSetDropoutDescriptor(dropout_descr,
            cudnn_handle, probability,
            states, state_size, seed);
    check_cudnn(cudnn_status);

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;

    cudnnTensorDescriptor_t x_descr;
    cudnn_status = cudnnCreateTensorDescriptor(&x_descr);
    check_cudnn(cudnn_status);
    cudnn_status = cudnnSetTensor4dDescriptor(x_descr,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, 1, X.rows, X.columns);
    check_cudnn(cudnn_status);
    cudnnTensorDescriptor_t y_descr;
    cudnn_status = cudnnCreateTensorDescriptor(&y_descr);
    check_cudnn(cudnn_status);
    cudnn_status = cudnnSetTensor4dDescriptor(y_descr,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, 1, Y.rows, Y.columns);
    check_cudnn(cudnn_status);
    void *d_X, *d_Y;
    // cudaMemcpy
    cuda_error = cudaMalloc(&d_X, X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X.values, X.rows * X.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cuda_error = cudaMalloc(&d_Y, Y.rows * Y.columns * sizeof(float));
    check_cuda(cuda_error);
    void *reserve_space;
    size_t reserve_space_size;
    cudnn_status = cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size);
    check_cudnn(cudnn_status);
    cuda_error = cudaMalloc(&reserve_space, reserve_space_size);
    check_cuda(cuda_error);
    cudnn_status = cudnnDropoutForward(cudnn_handle,
            dropout_descr, x_descr, d_X,
            y_descr, d_Y,
            reserve_space, reserve_space_size);
    check_cudnn(cudnn_status);

    Y.values = (float *) malloc(Y.rows * Y.columns * sizeof(float));
    cuda_error = cudaMemcpy(Y.values, d_Y, Y.rows * Y.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

    cudnn_status = cudnnDestroy(cudnn_handle);

    cuda_error = cudaFree(states);
    check_cuda(cuda_error);
    cuda_error = cudaFree(reserve_space);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_X);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_Y);
    check_cuda(cuda_error);

    return Y;
}


matrix<float> linear(matrix<float> X) {
    int num_hidden_channels = 8;
    matrix<float> weight;
    weight.rows = X.columns;
    weight.columns = num_hidden_channels;
    weight.values = (float *) malloc(weight.rows * weight.columns * sizeof(float));
    vector<float> bias;
    bias.size = num_hidden_channels;
    bias.values = (float *) malloc(bias.size * sizeof(float));

    // init weight and bias
    double k = 1.0 / (double) X.columns;
    k = sqrt(k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight.rows * weight.columns; ++i) {
        weight.values[i] = distr(gen);
    }
    for (int i = 0; i < bias.size; ++i) {
        bias.values[i] = distr(gen);
    }

    cudaError_t cuda_error;
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;
    cublas_status = cublasCreate(&cublas_handle);
    check_cublas(cublas_status);

    float *d_X, *d_weight, *d_bias;
    matrix<float> X_col;
    X_col.rows = X.columns;
    X_col.columns = X.rows;
    X_col.values = (float *) malloc(X_col.rows * X_col.columns * sizeof(float));
    transpose(X_col.values, X.values, X.rows, X.columns);
    cuda_error = cudaMalloc((void **) &d_X, X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X_col.values, X_col.rows * X_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    matrix<float> weight_col;
    weight_col.rows = weight.columns;
    weight_col.columns = weight.rows;
    weight_col.values = (float *) malloc(weight_col.rows * weight_col.columns * sizeof(float));
    transpose(weight_col.values, weight.values, weight.rows, weight.columns);
    cuda_error = cudaMalloc(&d_weight, weight_col.rows * weight_col.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_weight, weight_col.values,
            weight_col.rows * weight_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    matrix<float> bias_expanded;
    bias_expanded.rows = X.rows;
    bias_expanded.columns = bias.size;
    bias_expanded.values = (float *) malloc(bias_expanded.rows * bias_expanded.columns * sizeof(float));
    for (int i = 0; i < X.rows; ++i) {
        std::memcpy(&bias_expanded.values[i * bias.size],
                bias.values,
                bias.size * sizeof(float));
    }
    matrix<float> bias_expanded_col;
    bias_expanded_col.rows = bias_expanded.columns;
    bias_expanded_col.columns = bias_expanded.rows;
    bias_expanded_col.values = (float *) malloc(bias_expanded_col.rows * bias_expanded_col.columns * sizeof(float));
    transpose(bias_expanded_col.values, bias_expanded.values,
            bias_expanded.rows, bias_expanded.columns);
    cuda_error = cudaMalloc(&d_bias,
            bias_expanded_col.rows * bias_expanded_col.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_bias, bias_expanded_col.values,
            bias_expanded_col.rows * bias_expanded_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    
    float alpha = 1.0;
    float beta = 1.0;
    // PyTorch uses GEMM too
    cublas_status = cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            X.rows, num_hidden_channels, X.columns,
            &alpha,
            d_X, X.rows,
            d_weight, weight.rows,
            &beta,
            d_bias, X.rows);
    check_cublas(cublas_status);

    // get result of linear
    matrix<float> result_col;
    result_col.rows = bias_expanded_col.rows;
    result_col.columns = bias_expanded_col.columns;
    result_col.values = (float *) malloc(result_col.rows * result_col.columns * sizeof(float));
    cuda_error = cudaMemcpy(result_col.values, d_bias,
            result_col.rows * result_col.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);
    matrix<float> result;
    result.rows = result_col.columns;
    result.columns = result_col.rows;
    result.values = (float *) malloc(result.rows * result.columns * sizeof(float));
    transpose(result.values, result_col.values, result_col.rows, result_col.columns);

    // free GPU memory
    cuda_error = cudaFree(d_X);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_weight);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_bias);
    check_cuda(cuda_error);

    // free CPU memory
    free(weight.values);
    free(bias.values);
    free(X_col.values);
    free(weight_col.values);
    free(bias_expanded.values);
    free(bias_expanded_col.values);
    free(result_col.values);
    
    // clean cuBLAS
    cublas_status = cublasDestroy(cublas_handle);

    return result;
}

matrix<float> cudnn_forward(matrix<float> X, char mode) {
    char relu_mode = 'r';
    char softmax_mode = 's';
    cudaError_t cuda_error;
    cudnnStatus_t cudnn_status;
    cudnnHandle_t cudnn_handle;
    cudnn_status = cudnnCreate(&cudnn_handle);
    check_cudnn(cudnn_status);

    float *d_X;
    cuda_error = cudaMalloc(&d_X, X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X.values,
            X.rows * X.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cudnnTensorDescriptor_t x_desc;
    cudnn_status = cudnnCreateTensorDescriptor(&x_desc);
    check_cudnn(cudnn_status);
    if (mode == relu_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(x_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1, 1, X.rows, X.columns);
    } else if (mode == softmax_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(x_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                X.rows, 1, 1, X.columns);
    }
    check_cudnn(cudnn_status);
    
    matrix<float> result;
    result.rows = X.rows;
    result.columns = X.columns;
    result.values = (float *) malloc(result.rows * result.columns * sizeof(float));
    for (int i = 0; i < result.rows * result.columns; ++i) {
        result.values[i] = 0.0;
    }
    float *d_result;
    cuda_error = cudaMalloc(&d_result, result.rows * result.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_result, result.values,
            result.rows * result.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cudnnTensorDescriptor_t result_desc;
    cudnn_status = cudnnCreateTensorDescriptor(&result_desc);
    check_cudnn(cudnn_status);
    if (mode == relu_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(result_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1, 1, result.rows, result.columns);
    } else if (mode == softmax_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(result_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                result.rows, 1, 1, result.columns);
    }
    check_cudnn(cudnn_status);

    float alpha = 1.0;
    float beta = 0.0;
    if (mode == relu_mode) {
        cudnnActivationDescriptor_t relu_desc;
        cudnn_status = cudnnCreateActivationDescriptor(&relu_desc);
        check_cudnn(cudnn_status);
        double coef = std::numeric_limits<double>::max();
        cudnn_status = cudnnSetActivationDescriptor(relu_desc,
                CUDNN_ACTIVATION_RELU,
                CUDNN_PROPAGATE_NAN,
                coef);
        check_cudnn(cudnn_status);
    
        cudnn_status = cudnnActivationForward(cudnn_handle,
                relu_desc,
                &alpha, x_desc, d_X,
                &beta, result_desc, d_result);
        check_cudnn(cudnn_status);
    } else if (mode == softmax_mode) {
        cudnn_status = cudnnSoftmaxForward(cudnn_handle,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, x_desc, d_X,
                &beta, result_desc, d_result);
        check_cudnn(cudnn_status);
    }

    cuda_error = cudaMemcpy(result.values, d_result,
            result.rows * result.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

    // free GPU memory
    cuda_error = cudaFree(d_X);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_result);
    check_cuda(cuda_error);

    // clean cudnn
    cudnn_status = cudnnDestroy(cudnn_handle);
    check_cudnn(cudnn_status);

    return result;
}

matrix<float> relu(matrix<float> X) {
    return cudnn_forward(X, 'r');
}

matrix<float> softmax(matrix<float> X) {
    return cudnn_forward(X, 's');
}

float negative_log_likelihood_loss(matrix<float> X, vector<int> y) {
    double loss = 0.0;
    for (int i = 0; i < X.rows; ++i) {
        loss = loss + X.values[i * X.columns + y.values[i]];
    }
    loss = loss / X.columns;
    loss = - loss;
    return static_cast<float>(loss);
}


int main() {
    // read tensors
    // set path to directory
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";

    // read features
    std::string path = dir_path + "/features.npy";
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)) {
        std::cout << "features wrong data type" << std::endl;
    }
    float* arr_data_f = arr.data<float>();
    matrix<float> features;
    features.rows = arr.shape[0];
    features.columns = arr.shape[1];
    features.values = (float*) malloc(features.rows * features.columns * sizeof(float)); 
    std::memcpy(features.values, arr_data_f, features.rows * features.columns * sizeof(float));

    // read classes
    path = dir_path + "/classes.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(int)) {
        std::cout << "classes wrong data type" << std::endl;
    }
    int *arr_data_i = arr.data<int>();
    vector<int> classes;
    classes.size = arr.shape[0];
    classes.values = (int*) malloc(classes.size * sizeof(int)); 
    std::memcpy(classes.values, arr_data_i, classes.size * sizeof(int));

    // read train_mask
    path = dir_path + "/train_mask.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(bool)) {
        std::cout << "train_mask wrong data type" << std::endl;
    }
    bool *arr_data_b = arr.data<bool>();
    vector<bool> train_mask;
    train_mask.size = arr.shape[0];
    train_mask.values = (bool*) malloc(train_mask.size * sizeof(bool)); 
    std::memcpy(train_mask.values, arr_data_b, train_mask.size * sizeof(bool));

    // read val_mask
    path = dir_path + "/val_mask.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(bool)) {
        std::cout << "val_mask wrong data type" << std::endl;
    }
    arr_data_b = arr.data<bool>();
    vector<bool> val_mask;
    val_mask.size = arr.shape[0];
    val_mask.values = (bool*) malloc(val_mask.size * sizeof(bool)); 
    std::memcpy(val_mask.values, arr_data_b, val_mask.size * sizeof(bool));

    // read test_mask
    path = dir_path + "/test_mask.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(bool)) {
        std::cout << "test_mask wrong data type" << std::endl;
    }
    arr_data_b = arr.data<bool>();
    vector<bool> test_mask;
    test_mask.size = arr.shape[0];
    test_mask.values = reinterpret_cast<bool*>(malloc(test_mask.size * sizeof(bool)));
    std::memcpy(test_mask.values, arr_data_b, test_mask.size * sizeof(bool));

    // read adjacency
    path = dir_path + "/adjacency.mtx";
    char *path_char = &*path.begin();
    sparse_matrix<float> adjacency;
    int err = loadMMSparseMatrix<float>(path_char, 'f', true,
            &adjacency.rows, &adjacency.columns, &adjacency.nnz,
            &adjacency.csr_val, &adjacency.csr_row_ptr,
            &adjacency.csr_col_ind, true);
    if (err) {
        std::cout << "loadMMSparseMatrix failed" << std::endl;
    }
    one_to_zero_index(adjacency.csr_row_ptr, adjacency.rows + 1);
    one_to_zero_index(adjacency.csr_col_ind, adjacency.nnz);

    // graph convolution
    matrix<float> result = graph_convolution(adjacency, features, "sum");
    
    matrix<float> graph_conv_result_mean = graph_convolution(adjacency, features, "mean");

    // write result to npy file
    path = dir_path + "/graph_convolution_result.npy";
    std::vector<size_t> shape = {(size_t) result.rows, (size_t) result.columns};
    cnpy::npy_save<float>(path, result.values, shape);

    // dropout
    matrix<float> dropout_result = dropout(features);

    // write dropout result to npy file
    path = dir_path + "/dropout_result.npy";
    shape = {(size_t) dropout_result.rows, (size_t) dropout_result.columns};
    cnpy::npy_save<float>(path, dropout_result.values, shape);

    // linear layer
    matrix<float> linear_result = linear(features);

    // write linear layer result to npy file
    path = dir_path + "/linear_result.npy";
    shape = {(size_t) linear_result.rows, (size_t) linear_result.columns};
    cnpy::npy_save<float>(path, linear_result.values, shape);

    // ReLU
    matrix<float> relu_result = relu(features);

    // softmax
    matrix<float> softmax_result = softmax(linear_result);

    // loss
    float loss = negative_log_likelihood_loss(softmax_result, classes);

    std::cout << "loss " << loss << std::endl;

    // free memory
    free(features.values);
    free(classes.values);
    free(train_mask.values);
    free(val_mask.values);
    free(test_mask.values);
}

