// Copyright 2020 Marcel Wagenländer

#include <benchmark/benchmark.h>
#include <ctime>
#include <string>
#include <thread>


void alzheimer(std::string dataset);
void alzheimer_chunked(std::string dataset, int chunk_size);

const char kill_command[] = "pkill nvidia-smi";

void log_gpu(std::string name) {
    std::time_t start = std::time(0);
    std::string file_name = "/tmp/" + name + std::to_string(start) + ".smi";
    std::string command = "nvidia-smi dmon -s umt -o T -f " + file_name;
    system(command.c_str());
}

static void BM_Alzheimer_Flickr(benchmark::State &state) {
    std::string dataset = "flickr";
    std::thread gpu_logging(log_gpu, dataset);
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer(dataset);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Flickr);

static void BM_Alzheimer_Reddit(benchmark::State &state) {
    std::string dataset = "reddit";
    std::thread gpu_logging(log_gpu, dataset);
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer(dataset);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Reddit);

static void BM_Alzheimer_Chunked_Flickr_15(benchmark::State &state) {
    int power = 15;
    std::thread gpu_logging(log_gpu, "flickr_chunked_" + std::to_string(power) + "_");
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer_chunked("flickr", 1 << power);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Chunked_Flickr_15);

static void BM_Alzheimer_Chunked_Flickr_14(benchmark::State &state) {
    int power = 14;
    std::thread gpu_logging(log_gpu, "flickr_chunked_" + std::to_string(power) + "_");
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer_chunked("flickr", 1 << power);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Chunked_Flickr_14);

static void BM_Alzheimer_Chunked_Flickr_13(benchmark::State &state) {
    int power = 13;
    std::thread gpu_logging(log_gpu, "flickr_chunked_" + std::to_string(power) + "_");
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer_chunked("flickr", 1 << power);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Chunked_Flickr_13);

static void BM_Alzheimer_Chunked_Reddit_17(benchmark::State &state) {
    int power = 17;
    std::thread gpu_logging(log_gpu, "flickr_chunked_" + std::to_string(power) + "_");
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer_chunked("reddit", 1 << power);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Chunked_Reddit_17);

static void BM_Alzheimer_Chunked_Reddit_16(benchmark::State &state) {
    int power = 16;
    std::thread gpu_logging(log_gpu, "flickr_chunked_" + std::to_string(power) + "_");
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer_chunked("reddit", 1 << power);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Chunked_Reddit_16);

static void BM_Alzheimer_Chunked_Reddit_15(benchmark::State &state) {
    int power = 15;
    std::thread gpu_logging(log_gpu, "flickr_chunked_" + std::to_string(power) + "_");
    gpu_logging.detach();
    for (auto _ : state)
        alzheimer_chunked("reddit", 1 << power);
    system(kill_command);
}
BENCHMARK(BM_Alzheimer_Chunked_Reddit_15);

BENCHMARK_MAIN();
