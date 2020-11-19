// Copyright 2020 Marcel Wagenländer
#include <chrono>
#include <thread>


int main() {
    long num_elements = 1 << 31;
    long *array = (long *) malloc(num_elements * sizeof(long));
    for(long i = 0; i < num_elements; ++i) {
        array[i] = i;
    }
    std::this_thread::sleep_for(std::chrono::seconds(15));
    // free(array);
    array = (long *) malloc(num_elements * sizeof(long));
    for(long j = 0; j < num_elements; ++j) {
        array[j] = j;
    }
    std::this_thread::sleep_for(std::chrono::seconds(15));
    free(array);
}

