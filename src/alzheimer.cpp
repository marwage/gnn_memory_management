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
    // for (int i = 0; i < rows; i = i + 1) {
    for (int i = 0; i < 10; i = i + 1) {
        // for (int j = 0; j < cols; j = j + 1) {
        for (int j = 0; j < 10; j = j + 1) {
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

matrix<float> graph_convolution(sparse_matrix<float> A, matrix<float> B) {

    cudaError_t error = cudaSuccess;
    cusparseStatus_t sparse_status = CUSPARSE_STATUS_SUCCESS;
    cusparseHandle_t sparse_handle;
    sparse_status = cusparseCreate(&sparse_handle);
    check_cusparse(sparse_status);


    float *d_A_csr_val;
    int *d_A_csr_row_offsets, *d_A_col_ind;
    error = cudaMalloc((void**) &d_A_csr_val, 
            A.nnz * sizeof(float));
    check_cuda(error);
    error = cudaMalloc((void**) &d_A_csr_row_offsets,
            (A.rows + 1) * sizeof(int));
    check_cuda(error);
    error = cudaMalloc((void**) &d_A_col_ind,
            A.nnz * sizeof(int));
    check_cuda(error);
    error = cudaMemcpy(d_A_csr_val, A.csr_val,
            A.nnz * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda(error);
    error = cudaMemcpy(d_A_csr_row_offsets, A.csr_row_ptr,
            (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(error);
   error = cudaMemcpy(d_A_col_ind, A.csr_col_ind,
            A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda(error);
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
    error = cudaMalloc((void**) &d_B, B.rows * B.columns * sizeof(float));
    check_cuda(error);
    error = cudaMemcpy(d_B, B_col.values, B.rows * B.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(error);
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
    error = cudaMalloc((void**) &d_result, result.rows * result.columns * sizeof(float));
    check_cuda(error);
    error = cudaMemcpy(d_result, result_col.values, result_col.rows * result_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(error);
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
    error = cudaMalloc(&d_buffer, buffer_size);
    check_cuda(error);

    // compute SpMM
    sparse_status = cusparseSpMM(sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A_descr, B_descr, &beta, result_descr,
            CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT,
            d_buffer);
    check_cusparse(sparse_status);

    // move result_col to CPU memory
    error = cudaMemcpy(result_col.values, d_result, result_col.rows * result_col.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(error);

    // result to row-major
    transpose<float>(result.values, result_col.values, result_col.rows, result_col.columns);

    // free memory
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

    sparse_status = cusparseDestroy(sparse_handle);
    check_cusparse(sparse_status);

    return result;

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
    test_mask.values = (bool*) malloc(test_mask.size * sizeof(bool)); 
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
    matrix<float> result = graph_convolution(adjacency, features);

    // print result
    std::cout << "result" << std::endl;
    print_matrix(result.values, result.rows, result.columns);

    // write result to npy file
    path = dir_path + "/result.npy";
    std::vector<size_t> shape = {(size_t) result.rows, (size_t) result.columns};
    cnpy::npy_save<float>(path, result.values, shape);

    // free memory
    free(features.values);
    free(classes.values);
    free(train_mask.values);
    free(val_mask.values);
    free(test_mask.values);
}

