// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "gpu_memory_logger.hpp"

#include <catch2/catch.hpp>
#include <thread>
#include <chrono>


int measure_cuda_helper() {
    CudaHelper *helper;

    GPUMemoryLogger memory_logger("cuda_helper");
    memory_logger.start();

    long iterations = 5;
    for (long i = 0; i < iterations; ++i) {
        helper = new CudaHelper();

        std::this_thread::sleep_for(std::chrono::seconds(10));

        delete helper;
    }

    memory_logger.stop();

    return 1;
}

TEST_CASE("Cuda helper, memory", "[cudahelper][memory]") {
    CHECK(measure_cuda_helper());
}