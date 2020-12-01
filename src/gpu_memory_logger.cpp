// 2020 Marcel Wagenländer

#include "gpu_memory_logger.hpp"
#include "gpu_memory.hpp"


GPUMemoryLogger::GPUMemoryLogger(std::string file_name) {
    file_name_ = file_name;
    path_ = "/tmp/benchmark/" + file_name_ + ".log";
}

void GPUMemoryLogger::start() {
    std::future<void> future = signal_exit_.get_future();
//    std::thread logging_thread_(log_memory, path_, std::move(future));
    logging_thread_ = std::thread(log_memory, path_, std::move(future));
}

void GPUMemoryLogger::stop() {
    signal_exit_.set_value();
    logging_thread_.join();
}
