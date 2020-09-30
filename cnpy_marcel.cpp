#include "cnpy.h"


int main() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr";
    std::string path = dir_path + "/features.npy";
    cnpy::NpyArray arr = cnpy::npy_load(path);
    float* loaded_data = arr.data<float>();
    for (int i = 0; i < arr.shape.size(); i = i + 1) {
        std::cout << "shape " << i << " " << arr.shape[i] << "\n";
    }
    for (int i = 0; i < 10; i = i + 1) {
        std::cout << loaded_data[i] << "\n";
    }
}

