// 2020 Marcel Wagenländer

#include <iostream>


int main() {
    long num_elements = 5;
    
    long *a = new long[num_elements];
    for (long i = 0; i < num_elements; ++i) {
        a[i] = i + 1;
    }

    long *b = new long[num_elements];
    std::copy(a, a + num_elements, b);

    for (long i = 0; i < num_elements; ++i) {
        std::cout << b[i] << std::endl;
    }
}

