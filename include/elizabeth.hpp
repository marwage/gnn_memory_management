// 2020 Marcel Wagenländer

#ifndef ALZHEIMER_ALZHEIMER_H
#define ALZHEIMER_ALZHEIMER_H

#include "dataset.hpp"

#include <string>


void alzheimer(Dataset dataset);

void alzheimer_chunked(Dataset dataset, long chunk_size, bool keep_allocation);

void alzheimer_pipelined(Dataset dataset, long chunk_size);

#endif//ALZHEIMER_ALZHEIMER_H
