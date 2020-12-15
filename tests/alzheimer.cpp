// Copyright 2020 Marcel Wagenländer

#include "adam.hpp"
#include "add.hpp"
#include "cuda_helper.hpp"
#include "dropout.hpp"
#include "graph_convolution.hpp"
#include "helper.hpp"
#include "loss.hpp"
#include "relu.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <iostream>


int test_alzheimer(std::string dataset, int chunk_size) {
    throw "Not implemented";
}

TEST_CASE("Alzheimer", "[alzheimer]") {
    CHECK(test_alzheimer("flickr", 0));
    CHECK(test_alzheimer("products", 65536));
}
