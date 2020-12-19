// Copyright 2020 Marcel Wagenländer

#include "dataset.hpp"

std::string get_dataset_name(Dataset dataset) {
    if (dataset == flickr) {
        return "flickr";
    } else if (dataset == reddit) {
        return "reddit";
    } else if (dataset == products) {
        return "products";
    } else {
        return "";
    }
}
