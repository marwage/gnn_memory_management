// Copyright 2020 Marcel Wagenländer

#include <iostream>
#include <vector>


int main() {
    int N = 10;
    std::vector<int> numbers;
    numbers = std::vector<int>(N);
    for (int i = 0; i < N; ++i) {
        numbers[i] = i;
    }
    for (int i = 0; i < N; ++i) {
        std::cout << "Number " << i << ": " << numbers[i] << std::endl;
    }
}
