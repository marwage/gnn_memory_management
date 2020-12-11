// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "gpu_memory_logger.hpp"
#include "tensors.hpp"
#include "sparse_computation.hpp"

#include <benchmark/benchmark.h>


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string reddit_dir_path = dir_path + "/reddit";
const std::string products_dir_path = dir_path + "/products";


static void BM_OP_Transpose_CSR_Reddit(benchmark::State &state) {
    std::string path;
    CudaHelper cuda_helper;

    path = reddit_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    GPUMemoryLogger memory_logger("transpose_csr_reddit");
    memory_logger.start();

    for (auto _ : state) {
        transpose_csr_matrix(&adjacency, &cuda_helper);
    }

    memory_logger.stop();
}
BENCHMARK(BM_OP_Transpose_CSR_Reddit);

static void BM_OP_Transpose_CSR_Reddit_Chunked(benchmark::State &state) {
    std::string path;
    CudaHelper cuda_helper;
    long chunk_size = state.range(0);

    path = reddit_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    SparseMatrix<float> adjacency_chunk;
    get_rows(&adjacency_chunk, &adjacency, 0, chunk_size);

    GPUMemoryLogger memory_logger("transpose_csr_reddit_" + std::to_string(chunk_size));
    memory_logger.start();

    for (auto _ : state) {
        transpose_csr_matrix(&adjacency_chunk, &cuda_helper);
    }

    memory_logger.stop();
}
BENCHMARK(BM_OP_Transpose_CSR_Reddit_Chunked)->Range(1 << 12, 1 << 17);