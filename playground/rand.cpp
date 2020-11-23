include <cmath>
#include <iostream>
#include <random>
#include <chrono>


int main() {
    long num_in_features = 512;
    double k = 1.0 / static_cast<double>(num_in_features);
    k = sqrt(k);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < 5; ++i) {
        std::cout << distr(generator) << std::endl;
    }
}

