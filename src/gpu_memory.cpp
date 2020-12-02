// Copyright 2020 Marcel Wagenländer

#include "gpu_memory.hpp"
#include "cuda_helper.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

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

void log_memory(std::string path, std::future<void> future) {
    std::chrono::high_resolution_clock::time_point tp_start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point tp_now;
    while (future.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        tp_now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_span = tp_now - tp_start;
        log_allocated_memory(path, time_span.count());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}