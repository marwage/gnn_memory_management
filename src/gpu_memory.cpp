// Copyright 2020 Marcel Wagenländer

#include "gpu_memory.hpp"
#include "cuda_helper.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

#define MiB (1 << 20)


// return is in Bytes
long get_allocated_memory() {
    size_t free;
    size_t total;
    check_cuda(cudaMemGetInfo(&free, &total));
    return total - free;
}

void print_allocated_memory(std::string name) {
    std::cout << name << ": Allocated memory: " << (get_allocated_memory() / MiB) << " MiB" << std::endl;
}

void log_allocated_memory(std::string path, double time_point) {
    std::ofstream log_file;
    log_file.open(path, std::ios::app);
    log_file << time_point << "," << (get_allocated_memory() / MiB) << "\n";
    log_file.close();
}
