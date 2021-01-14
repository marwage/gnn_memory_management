#include <iostream>
#include <thread>


void fac(long *res, long num) {
    if (num > 0) {
        *res = *res * num;
        std::thread thr(fac, res, num - 1);
        thr.join();
    }
}


int main() {
    long res = 1;
    std::thread first(fac, &res, 10);
    first.join();
    std::cout << "Result " << res << std::endl;
}
