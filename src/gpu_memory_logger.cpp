// 2020 Marcel Wagenländer

#include "gpu_memory_logger.hpp"
#include "gpu_memory.hpp"

#define MiB (1 << 20)


void log_memory(std::future<void> future, std::string *log_string, long interval) {
    std::chrono::high_resolution_clock::time_point tp_start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point tp_now;
    while (future.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        tp_now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_span = tp_now - tp_start;

        long allocated_memory_mib = get_allocated_memory() / MiB;

        log_string->append(std::to_string(time_span.count()) + "," + std::to_string(allocated_memory_mib) + "\n");

        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
}

GPUMemoryLogger::GPUMemoryLogger(std::string file_name) {
    file_name_ = file_name;
    path_ = "/tmp/benchmark/" + file_name_ + ".log";
    interval_ = 100;
}

GPUMemoryLogger::GPUMemoryLogger(std::string file_name, long interval) {
    file_name_ = file_name;
    path_ = "/tmp/benchmark/" + file_name_ + ".log";
    interval_ = interval;
    log_string_ = "time,memory\n";
}

void GPUMemoryLogger::start() {
    std::future<void> future = signal_exit_.get_future();
    logging_thread_ = std::thread(log_memory, std::move(future), &log_string_, interval_);
}

void GPUMemoryLogger::stop() {
    signal_exit_.set_value();
    logging_thread_.join();

    log_file_.open(path_, std::ios::trunc);
    log_file_ << log_string_;
    log_file_.close();
}
