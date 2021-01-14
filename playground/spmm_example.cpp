#include "cnpy.h"
#include "mmio_wrapper.h"

#include "cusparse.h"
#include <cuda_runtime.h>


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

template<typename T>
void print_matrix(T *a, int rows, int cols) {
    for (int i = 0; i < rows; i = i + 1) {
        for (int j = 0; j < cols; j = j + 1) {
            std::cout << a[i * cols + j] << ",";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_vector(T *a, int num_ele) {
    print_matrix(a, num_ele, 1);
}

int new_index(int old_idx, int N, int M) {
    int last_idx = M * N - 1;
    if (old_idx == last_idx) {
        return last_idx;
    } else {
        return (N * old_idx) % last_idx;
    }
}

template<typename T>
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


int main() {
    // test transpose
    // int N = 2;
    // int M = 5;
    // float X[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    // std::cout << "X" << std::endl;
    // print_matrix(X, N, M);

    // float X_T[N * M];
    // transpose(X_T, X, N, M);
    // std::cout << "X transposed" << std::endl;
    // print_matrix(X_T, M, N);


    // read tensors
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/example";

    std::string path = dir_path + "/B.npy";
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)) {
        std::cout << "features wrong data type" << std::endl;
    }
    float *arr_data_f = arr.data<float>();
    int B_N = arr.shape[0];
    int B_M = arr.shape[1];
    float *B = (float *) malloc(B_N * B_M * sizeof(float));
    std::memcpy(B, arr_data_f, B_N * B_M * sizeof(float));
    // std::cout << "B" << std::endl;
    // print_matrix<float>(B, B_N, B_M);

    path = dir_path + "/A.mtx";
    char *path_char = &*path.begin();
    int A_rows, A_cols, A_nnz;
    float *A_csr_val;
    int *A_csr_row_ptr, *A_csr_col_ind;
    int err = loadMMSparseMatrix<float>(path_char, 'f', true,
                                        &A_rows, &A_cols, &A_nnz,
                                        &A_csr_val, &A_csr_row_ptr,
                                        &A_csr_col_ind, true);
    if (err) {
        std::cout << "loadMMSparseMatrix failed" << std::endl;
    }

    // to zero indexed
    for (int i = 0; i < A_nnz; ++i) {
        A_csr_col_ind[i] = A_csr_col_ind[i] - 1;
    }
    for (int i = 0; i < A_rows + 1; i++) {
        A_csr_row_ptr[i] = A_csr_row_ptr[i] - 1;
    }

    std::cout << "adjacency" << std::endl;
    std::cout << "rows " << A_rows << std::endl;
    std::cout << "columns " << A_cols << std::endl;
    std::cout << "nnz " << A_nnz << std::endl;
    for (int i = 0; i < A_nnz; ++i) {
        std::cout << "val " << A_csr_val[i] << std::endl;
    }
    for (int i = 0; i < A_nnz; ++i) {
        std::cout << "col " << A_csr_col_ind[i] << std::endl;
    }
    for (int i = 0; i < A_rows + 1; i++) {
        std::cout << "row " << A_csr_row_ptr[i] << std::endl;
    }

    // compute graph convolution
    cudaError_t error = cudaSuccess;
    cusparseStatus_t sparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparseHandle_t sparse_handle;
    sparse_status = cusparseCreate(&sparse_handle);
    check_cusparse(sparse_status);

    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    error = cudaMalloc((void **) &d_A_csr_val,
                       A_nnz * sizeof(float));
    check_cuda(error);
    error = cudaMalloc((void **) &d_A_csr_row_offsets,
                       (A_rows + 1) * sizeof(int));
    check_cuda(error);
    error = cudaMalloc((void **) &d_A_col_ind,
                       A_nnz * sizeof(int));
    check_cuda(error);
    error = cudaMemcpy(d_A_csr_val, A_csr_val,
                       A_nnz * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(error);
    error = cudaMemcpy(d_A_csr_row_offsets, A_csr_row_ptr,
                       (A_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(error);
    error = cudaMemcpy(d_A_col_ind, A_csr_col_ind,
                       A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(error);

    cusparseSpMatDescr_t A_descr;
    sparse_status = cusparseCreateCsr(&A_descr, A_rows,
                                      A_cols, A_nnz,
                                      d_A_csr_row_offsets, d_A_col_ind,
                                      d_A_csr_val,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    check_cusparse(sparse_status);

    float B_T[B_N * B_M];
    transpose<float>(B_T, B, B_N, B_M);// to column-major

    std::cout << "B" << std::endl;
    print_matrix(B, B_N, B_M);
    std::cout << "B column-major" << std::endl;
    print_matrix(B_T, B_M, B_N);

    float *d_B;
    error = cudaMalloc((void **) &d_B, B_N * B_M * sizeof(float));
    check_cuda(error);
    error = cudaMemcpy(d_B, B_T, B_N * B_M * sizeof(float),
                       cudaMemcpyHostToDevice);
    check_cuda(error);

    cusparseDnMatDescr_t B_descr;
    sparse_status = cusparseCreateDnMat(&B_descr, B_N, B_M,
                                        B_N, d_B,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    check_cusparse(sparse_status);

    float *result = (float *) malloc(A_rows * B_M * sizeof(float));
    for (int i = 0; i < A_rows * B_M; ++i) {
        result[i] = 0.0f;
    }
    float *d_result;
    error = cudaMalloc((void **) &d_result, A_rows * B_M * sizeof(float));
    check_cuda(error);
    error = cudaMemcpy(d_result, result, A_rows * B_M * sizeof(float),
                       cudaMemcpyHostToDevice);
    check_cuda(error);

    cusparseDnMatDescr_t result_descr;
    sparse_status = cusparseCreateDnMat(&result_descr, A_rows, B_M,
                                        A_rows, d_result,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    check_cusparse(sparse_status);

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
    error = cudaMalloc(&d_buffer, buffer_size);
    check_cuda(error);

    sparse_status = cusparseSpMM(sparse_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, A_descr, B_descr, &beta, result_descr,
                                 CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
                                 d_buffer);
    check_cusparse(sparse_status);

    error = cudaMemcpy(result, d_result, A_rows * B_M * sizeof(float),
                       cudaMemcpyDeviceToHost);
    check_cuda(error);

    std::cout << "result columns-major" << std::endl;
    print_matrix(result, B_M, A_rows);

    float result_T[A_rows * B_M];
    transpose<float>(result_T, result, B_M, A_rows);
    std::cout << "result" << std::endl;
    print_matrix(result_T, A_rows, B_M);

    path = dir_path + "/result.npy";
    std::vector<size_t> shape = {(size_t) A_rows, (size_t) B_M};
    cnpy::npy_save<float>(path, result_T, shape);

    error = cudaFree(d_A_csr_val);
    check_cuda(error);
    error = cudaFree(d_A_col_ind);
    check_cuda(error);
    error = cudaFree(d_A_csr_row_offsets);
    check_cuda(error);
    error = cudaFree(d_B);
    check_cuda(error);
    error = cudaFree(d_buffer);
    check_cuda(error);
    error = cudaFree(d_result);
    check_cuda(error);

    free(B);
    free(result);

    sparse_status = cusparseDestroy(sparse_handle);
    check_cusparse(sparse_status);
}
