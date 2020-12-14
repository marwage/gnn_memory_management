// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_PIPELINE_H
#define ALZHEIMER_PIPELINE_H

#include "layer.hpp"


class LayerPipeline {
public:
    long num_chunks_;

    void in(long chunk, long buffer);
    void out(long chunk, long buffer);
    void compute(long buffer);
};

void pipeline(Layer *layer);

#endif//ALZHEIMER_PIPELINE_H
