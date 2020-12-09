// 2020 Marcel Wagenländer

#ifndef ALZHEIMER_GPU_MEMORY_LOGGER_H
#define ALZHEIMER_GPU_MEMORY_LOGGER_H

#include <future>
#include <string>
#include <thread>
#include <fstream>


class GPUMemoryLogger {
private:
    std::string file_name_;
    std::string path_;
    long interval_;
    std::promise<void> signal_exit_;
    std::thread logging_thread_;
    std::ofstream log_file_;

public:
    GPUMemoryLogger(std::string file_name, long interval);
    GPUMemoryLogger(std::string file_name);
    void start();
    void stop();
};

#endif//ALZHEIMER_GPU_MEMORY_LOGGER_H
