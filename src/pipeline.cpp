// Copyright 2020 Marcel Wagenländer

#include "pipeline.hpp"


void pipeline(LayerPipeline *layer) {
    long chunk_zero = 0;
    long chunk_one = 1;
    long chunk_two = 2;
    for (int i = 0; i < layer->num_chunks_ + 2; ++i) {
        // update chunk offsets
        chunk_zero = (i / 3) * 3; // every three steps jump by 3
        chunk_one = ((i - 1) / 3)  * 3 + 1; // one tick behind and one number higher
        chunk_two = ((i - 2) / 3)  * 3 + 2; // two ticks behind and two number higher

        if (i % 3 == 0) {
            // zero in, one out, two compute
            // zero in
            if (chunk_zero < layer->num_chunks_) {
                layer->in(chunk_zero, 0);
            }

            // one out
            if (chunk_one < layer->num_chunks_ && i > 2) {
                layer->out(chunk_one, 1);
            }

            // two computation
            if (chunk_two < layer->num_chunks_ && i > 2) {
                layer->compute(2);
            }
        } else if (i % 3 == 1) {
            // one in, two out, zero compute
            // one in
            if (chunk_one < layer->num_chunks_ && i > 0) {
                layer->in(chunk_one, 1);
            }

            // two out
            if (chunk_two < layer->num_chunks_ && i > 3) {
                layer->out(chunk_two, 2);
            }

            // zero compute
            if (chunk_zero < layer->num_chunks_ && i > 0) {
                layer->compute(0);
            }
        } else if (i % 3 == 2) {
            // two in, zero out, one compute
            // two in
            if (chunk_two < layer->num_chunks_ && i > 0) {
                layer->in(chunk_two, 2);
            }

            // zero out
            if (chunk_zero < layer->num_chunks_ && i > 1) {
                layer->out(chunk_zero, 0);
            }

            // one compute
            if (chunk_one < layer->num_chunks_ && i > 0) {
                layer->compute(1);
            }
        }

        // sync all spanned calls
        check_cuda(cudaDeviceSynchronize());
    }
}