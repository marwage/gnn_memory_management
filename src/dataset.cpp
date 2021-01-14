// Copyright 2020 Marcel Wagenländer

#include "dataset.hpp"

std::string get_dataset_name(Dataset dataset) {
    if (dataset == flickr) {
        return "flickr";
    } else if (dataset == reddit) {
        return "reddit";
    } else if (dataset == products) {
        return "products";
    } else if (dataset == ivy) {
        return "ivy";
    } else {
        throw "Unkown dataset";
    }
}

long get_dataset_num_classes(Dataset dataset) {
    if (dataset == flickr) {
        return 7;
    } else if (dataset == reddit) {
        return 41;
    } else if (dataset == products) {
        return 47;
    } else if (dataset == ivy) {
        return 64;
    } else {
        throw "Unkown dataset";
    }
}
