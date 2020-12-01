// 2020 Marcel Wagenländer

#ifndef ALZHEIMER_GPU_MEMORY_LOGGER_H
#define ALZHEIMER_GPU_MEMORY_LOGGER_H

#include <future>
#include <thread>
#include <string>


class GPUMemoryLogger {
private:
    std::string file_name_;
    std::string path_;
    std::promise<void> signal_exit_;
    std::thread logging_thread_;

public:
    GPUMemoryLogger(std::string file_name);
    void start();
    void stop();
};

#endif//ALZHEIMER_GPU_MEMORY_LOGGER_H
