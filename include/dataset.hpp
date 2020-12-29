// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_DATASET_HPP
#define ALZHEIMER_DATASET_HPP

#include <string>


enum Dataset { flickr,
               reddit,
               products,
               ivy };

std::string get_dataset_name(Dataset dataset);

#endif//ALZHEIMER_DATASET_HPP
