#include "string.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xnpy.hpp"
#include <cstdlib>


int main(int argc, char **argv) {
    const char *home = std::getenv("HOME");
    char *path;
    strcpy(path, home);
    strcat(path, "/gpu_memory_reduction/alzheimer/data/flickr/adj_full.npz");
    auto data = xt::load_npy<float>(path);

    return 0;
}
