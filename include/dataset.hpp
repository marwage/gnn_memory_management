// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_DATASET_HPP
#define ALZHEIMER_DATASET_HPP

#include <string>

enum Dataset { flickr,
               reddit,
               products,
               ivy };

struct DatasetStats {
    long num_nodes;
    long num_edges;
    long num_features;
    long num_classes;
};

std::string get_dataset_name(Dataset dataset);

long get_dataset_num_classes(Dataset dataset);

DatasetStats get_dataset_stats(Dataset dataset);

#endif//ALZHEIMER_DATASET_HPP
