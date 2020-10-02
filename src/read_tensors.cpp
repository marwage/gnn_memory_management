#include "cnpy.h"
#include "mmio_wrapper.h"


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
    for (int i = 0; i < num_ele; i = i + 1) {
        std::cout << a[i] << ",";
    }
    std::cout << std::endl;
}

int main() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";
    
    std::string path = dir_path + "/features.npy";
    cnpy::NpyArray arr = cnpy::npy_load(path);
    float* features = arr.data<float>();
    if (arr.word_size != sizeof(float)) {
        std::cout << "features wrong data type" << std::endl;
    }
    // print_matrix<float>(features, arr.shape[0], arr.shape[1]);

    path = dir_path + "/classes.npy";
    arr = cnpy::npy_load(path);
    int* classes = arr.data<int>();
    if (arr.word_size != sizeof(int)) {
        std::cout << "classes wrong data type" << std::endl;
    }
    // print_matrix<int>(classes, arr.shape[0], arr.shape[1]);

    path = dir_path + "/train_mask.npy";
    arr = cnpy::npy_load(path);
    bool* train_mask = arr.data<bool>();
    if (arr.word_size != sizeof(bool)) {
        std::cout << "train_mask wrong data type" << std::endl;
    }
    // print_vector<bool>(train_mask, arr.shape[0]);

    path = dir_path + "/val_mask.npy";
    arr = cnpy::npy_load(path);
    bool* val_mask = arr.data<bool>();
    if (arr.word_size != sizeof(bool)) {
        std::cout << "val_mask wrong data type" << std::endl;
    }
    // print_vector<bool>(val_mask, arr.shape[0]);

    path = dir_path + "/test_mask.npy";
    arr = cnpy::npy_load(path);
    bool* test_mask = arr.data<bool>();
    if (arr.word_size != sizeof(bool)) {
        std::cout << "test_mask wrong data type" << std::endl;
    }
    // print_vector<bool>(test_mask, arr.shape[0]);

    path = dir_path + "/adjacency.mtx";
    char *path_char = &*path.begin();
    int adjacency_rows, adjacency_cols, adjacency_nnz;
    float *adjacency_csr_val;
    int *adjacency_csr_row_ptr, *adjacency_csr_col_ind;
    loadMMSparseMatrix<float>(path_char, 'f', true, &adjacency_rows,
        &adjacency_cols, &adjacency_nnz, &adjacency_csr_val, &adjacency_csr_row_ptr,
        &adjacency_csr_col_ind, true);
}

