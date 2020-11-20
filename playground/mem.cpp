// Copyright 2020 Marcel Wagenländer
#include <chrono>
#include <thread>
#include <iostream>


void overwrite() {
    long num_elements = 1 << 30;
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

void free_pointer() {
    std::cout << "Size of long " << sizeof(long) << std::endl;
    long num_elements = 1 << 30;
    std::cout << "Number of elements " << num_elements << std::endl;
    long *array = (long *) malloc(num_elements * sizeof(long));
    for(long i = 0; i < num_elements; ++i) {
        array[i] = i + 1;
    }
    std::cout << "Array pointer " << array << std::endl;
    std::cout << "Array first element " << array[0] << std::endl;
    free(array);
    array = NULL;
    std::cout << "Array pointer after free " << array << std::endl;
}

int main() {
    free_pointer();
}

