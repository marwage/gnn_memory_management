#include "cnpy.h"
#include "mmio_wrapper.h"

#include <cuda_runtime.h>
#include "cusparse.h"


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

template <typename T>
void print_matrix(T* a, int rows, int cols) {
    for (int i = 0; i < rows; i = i + 1) {
        for (int j = 0; j < cols; j = j + 1) {
            std::cout << a[i * cols + j] << ",";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print_vector(T* a, int num_ele) {
    print_matrix(a, num_ele, 1);
}

int main() {
    // read tensors
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";

    std::string path = dir_path + "/features.npy";
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)) {
        std::cout << "features wrong data type" << std::endl;
    }
    float* arr_data_f = arr.data<float>();
    int features_N = arr.shape[0];
    int features_M = arr.shape[1];
    float *features = (float*) malloc(features_N * features_M * sizeof(float)); 
    std::memcpy(features, arr_data_f, features_N * features_M * sizeof(float));
    // print_matrix<float>(features, features_N, features_M);

    path = dir_path + "/classes.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(int)) {
        std::cout << "classes wrong data type" << std::endl;
    }
    int *arr_data_i = arr.data<int>();
    int classes_N = arr.shape[0];
    int *classes = (int*) malloc(classes_N * sizeof(int)); 
    std::memcpy(classes, arr_data_i, classes_N * sizeof(int));
    // print_vector<int>(classes, classes_N);

    path = dir_path + "/train_mask.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(bool)) {
        std::cout << "train_mask wrong data type" << std::endl;
    }
    bool *arr_data_b = arr.data<bool>();
    int train_mask_N = arr.shape[0];
    bool *train_mask = (bool*) malloc(train_mask_N * sizeof(float)); 
    std::memcpy(train_mask, arr_data_b, train_mask_N * sizeof(float));
    // print_vector<bool>(train_mask, train_mask_N);

    path = dir_path + "/val_mask.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(bool)) {
        std::cout << "val_mask wrong data type" << std::endl;
    }
    arr_data_b = arr.data<bool>();
    int val_mask_N = arr.shape[0];
    bool *val_mask = (bool*) malloc(val_mask_N * sizeof(float)); 
    std::memcpy(val_mask, arr_data_b, val_mask_N * sizeof(float));
    // print_vector<bool>(val_mask, val_mask_N);

    path = dir_path + "/test_mask.npy";
    arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(bool)) {
        std::cout << "test_mask wrong data type" << std::endl;
    }
    arr_data_b = arr.data<bool>();
    int test_mask_N = arr.shape[0];
    bool *test_mask = (bool*) malloc(test_mask_N * sizeof(float)); 
    std::memcpy(test_mask, arr_data_b, test_mask_N * sizeof(float));
    // print_vector<bool>(test_mask, test_mask_N);

    path = dir_path + "/adjacency.mtx";
    char *path_char = &*path.begin();
    int adjacency_rows, adjacency_cols, adjacency_nnz;
    float *adjacency_csr_val;
    int *adjacency_csr_row_ptr, *adjacency_csr_col_ind;
    int err = loadMMSparseMatrix<float>(path_char, 'f', true,
            &adjacency_rows, &adjacency_cols, &adjacency_nnz,
            &adjacency_csr_val, &adjacency_csr_row_ptr,
            &adjacency_csr_col_ind, true);
    if (err) {
        std::cout << "loadMMSparseMatrix failed" << std::endl;
    }


    // compute graph convolution
    cudaError_t error = cudaSuccess;
    cusparseStatus_t sparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparseHandle_t sparse_handle;
    sparse_status = cusparseCreate(&sparse_handle);
    check_cusparse(sparse_status);

    float *d_adjacency_csr_val;
    int *d_adjacency_csr_row_offsets, *d_adjacency_col_ind;
    error = cudaMalloc((void**) &d_adjacency_csr_val, 
            adjacency_nnz * sizeof(float));
    check_cuda(error);
    error = cudaMalloc((void**) &d_adjacency_csr_row_offsets,
            (adjacency_nnz + 1) * sizeof(int));
    check_cuda(error);
    error = cudaMalloc((void**) &d_adjacency_col_ind,
            adjacency_nnz * sizeof(int));
    check_cuda(error);
    error = cudaMemcpy(d_adjacency_csr_val, adjacency_csr_val,
            adjacency_nnz * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(error);
    error = cudaMemcpy(d_adjacency_csr_row_offsets, adjacency_csr_row_ptr,
            (adjacency_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(error);
    error = cudaMemcpy(d_adjacency_col_ind, adjacency_csr_col_ind,
            adjacency_nnz * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(error);

    cusparseSpMatDescr_t adjacency_descr;
    sparse_status = cusparseCreateCsr(&adjacency_descr, adjacency_rows,
            adjacency_cols, adjacency_nnz,
            d_adjacency_csr_row_offsets, d_adjacency_col_ind,
            d_adjacency_csr_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    check_cusparse(sparse_status);
    
    float *d_features;
    error = cudaMalloc((void**) &d_features, features_N * features_M * sizeof(float));
    check_cuda(error);
    error = cudaMemcpy(d_features, features, features_N * features_M * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(error);
    
    cusparseDnMatDescr_t features_descr;
    sparse_status = cusparseCreateDnMat(&features_descr, features_N, features_M,
            features_N, d_features,
            CUDA_R_32F, CUSPARSE_ORDER_COL);
    check_cusparse(sparse_status);

    float *d_result;
    error = cudaMalloc((void**) &d_result, features_N * features_M * sizeof(float));
    check_cuda(error);

    cusparseDnMatDescr_t result_descr;
    sparse_status = cusparseCreateDnMat(&result_descr, features_N, features_M,
            features_N, d_features,
            CUDA_R_32F, CUSPARSE_ORDER_COL);
    check_cusparse(sparse_status);

    float alpha = 1.0f;
    float beta = 0.0f;
    size_t buffer_size;
    sparse_status = cusparseSpMM_bufferSize(sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, adjacency_descr, features_descr, &beta, result_descr,
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
            &alpha, adjacency_descr, features_descr, &beta, result_descr,
            CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
            d_buffer);
    check_cusparse(sparse_status);

    float* result = (float *) malloc(features_N * features_M * sizeof(float));
    error = cudaMemcpy(result, d_result, features_N * features_M * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(error);

    print_matrix(result, features_N, features_M);

    error = cudaFree(d_adjacency_csr_val);
    check_cuda(error);
    error = cudaFree(d_adjacency_col_ind);
    check_cuda(error);
    error = cudaFree(d_adjacency_csr_row_offsets);
    check_cuda(error);
    error = cudaFree(d_features);
    check_cuda(error);
    error = cudaFree(d_buffer);
    check_cuda(error);

    free(features);
    free(classes);
    free(train_mask);
    free(val_mask);
    free(test_mask);

    sparse_status = cusparseDestroy(sparse_handle);
    check_cusparse(sparse_status);
}

