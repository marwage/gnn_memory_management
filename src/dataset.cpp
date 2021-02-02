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
    DatasetStats dataset_stats = get_dataset_stats(dataset);

    return dataset_stats.num_classes;
}

DatasetStats get_dataset_stats(Dataset dataset) {
    DatasetStats dataset_stats;

    if (dataset == flickr) {
        dataset_stats.num_nodes = 89250;
        dataset_stats.num_edges = 899756;
        dataset_stats.num_features = 500;
        dataset_stats.num_classes = 7;
        return dataset_stats;
    } else if (dataset == reddit) {
        dataset_stats.num_nodes = 232965;
        dataset_stats.num_edges = 114615892;
        dataset_stats.num_features = 602;
        dataset_stats.num_classes = 41;
        return dataset_stats;
    } else if (dataset == products) {
        dataset_stats.num_nodes = 2449029;
        dataset_stats.num_edges = 61859140;
        dataset_stats.num_features = 100;
        dataset_stats.num_classes = 47;
        return dataset_stats;
    } else if (dataset == ivy) {
        dataset_stats.num_nodes = 8388608;
        dataset_stats.num_edges = 703687441;
        dataset_stats.num_features = 512;
        dataset_stats.num_classes = 64;
        return dataset_stats;
    } else {
        throw "Unkown dataset";
    }
}
