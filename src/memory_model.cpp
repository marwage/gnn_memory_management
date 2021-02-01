// Copyright 2021 Marcel Wagenländer

#include "memory_model.hpp"

#include <limits>

ChunkSizeEquation::ChunkSizeEquation(long max_num_elements) {
    a_ = 0;
    b_ = 0;
    max_num_elements_ = max_num_elements;
}

ChunkSizeEquation::ChunkSizeEquation(long max_num_elements, long a, long b) {
    max_num_elements_ = max_num_elements;
    ChunkSizeEquation::set_coefficients(a, b);
}

void ChunkSizeEquation::set_coefficients(long a, long b) {
    a_ = a;
    b_ = b;
}

void ChunkSizeEquation::add(long a, long b) {
    a_ = a_ + a;
    b_ = b_ + b;
}

long ChunkSizeEquation::get_chunk_size() {
    return (max_num_elements_ - b_) / a_;
}

// MEMORY MODEL

MemoryModel::~MemoryModel() {}

std::string MemoryModel::get_name() {
    return name_;
}

long MemoryModel::get_memory_usage() {
    return memory_usage_;
}

std::vector<long> MemoryModel::get_chunk_size_coefficients() {
    return {a_, b_};
}

long MemoryModel::get_max_chunk_size(long max_num_elements) {
    ChunkSizeEquation chunk_size_eq(max_num_elements, a_, b_);
    return chunk_size_eq.get_chunk_size();
}

// COMPOSITION MEMORY MODEL

CompositionMemoryModel::~CompositionMemoryModel() {
    for (MemoryModel *layer : layers_) {
        delete layer;
    }
}

long CompositionMemoryModel::get_memory_usage() {
    long memory_usage = 0;
    for (MemoryModel *layer : layers_) {
        long layer_memory_usage = layer->get_memory_usage();
        memory_usage = memory_usage + layer_memory_usage;
    }
    return memory_usage;
}

std::vector<long> CompositionMemoryModel::get_chunk_size_coefficients() {
    long a = 0;
    long b = 0;
    for (MemoryModel *layer : layers_) {
        std::vector<long> vars = layer->get_chunk_size_coefficients();
        a = a + vars.at(0);
        b = b + vars.at(1);
    }
    return {a, b};
}

long CompositionMemoryModel::get_max_chunk_size(long max_num_elements) {
    std::vector<long> coefficients = CompositionMemoryModel::get_chunk_size_coefficients();

    ChunkSizeEquation chunk_size_eq(max_num_elements, coefficients.at(0), coefficients.at(1));

    return chunk_size_eq.get_chunk_size();
}

ChunkSizeLayer CompositionMemoryModel::get_max_chunk_size_layer(long max_num_elements) {
    long max_chunk_size = std::numeric_limits<long>::max();
    MemoryModel *max_layer;

    for (MemoryModel *layer : layers_) {
        CompositionMemoryModel* composition_layer = dynamic_cast<CompositionMemoryModel *>(layer);
        if (composition_layer != NULL) {
            ChunkSizeLayer chunk_size_layer = composition_layer->get_max_chunk_size_layer(max_num_elements);
            if (chunk_size_layer.chunk_size < max_chunk_size) {
                max_chunk_size = chunk_size_layer.chunk_size;
                max_layer = chunk_size_layer.layer;
            }
        } else {
            long max_chunk_size_layer = layer->get_max_chunk_size(max_num_elements);
            if (max_chunk_size_layer < max_chunk_size) {
                max_chunk_size = max_chunk_size_layer;
                max_layer = layer;
            }
        }
    }

    ChunkSizeLayer chunk_size_layer;
    chunk_size_layer.layer = max_layer;
    chunk_size_layer.chunk_size = max_chunk_size;
    return chunk_size_layer;
}

MemoryUsageLayer CompositionMemoryModel::get_memory_usage_layer() {
    long max_memory_usage = 0;
    MemoryModel *max_layer;

    for (MemoryModel *layer : layers_) {
        CompositionMemoryModel* composition_layer = dynamic_cast<CompositionMemoryModel *>(layer);
        if (composition_layer != NULL) {
            MemoryUsageLayer composition_layer_memory = composition_layer->get_memory_usage_layer();
            if (composition_layer_memory.memory_usage > max_memory_usage) {
                max_memory_usage = composition_layer_memory.memory_usage;
                max_layer = composition_layer_memory.layer;
            }
        } else {
            long max_memory_usage_layer = layer->get_memory_usage();
            if (max_memory_usage_layer > max_memory_usage) {
                max_memory_usage = max_memory_usage_layer;
                max_layer = layer;
            }
        }
    }

    MemoryUsageLayer chunk_size_layer;
    chunk_size_layer.layer = max_layer;
    chunk_size_layer.memory_usage = max_memory_usage;
    return chunk_size_layer;
}

// LAYERS

DropoutMemoryModel::DropoutMemoryModel(long num_nodes, long num_features) {
    name_ = "Dropout";
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    memory_usage_ = 2 * num_nodes_ * num_features_;
    a_ = 2 * num_features;
    b_ = 0;
}

ReluMemoryModel::ReluMemoryModel(long num_nodes, long num_features) {
    name_ = "Relu";
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    memory_usage_ = 4 * num_nodes * num_features;
    a_ = 4 * num_features;
    b_ = 0;
}

LogSoftmaxMemoryModel::LogSoftmaxMemoryModel(long num_nodes, long num_features) {
    name_ = "Log-softmax";
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    memory_usage_ = 3 * num_nodes * num_features;
    a_ = 3 * num_features_;
    b_ = 0;
}

LinearMemoryModel::LinearMemoryModel(long num_nodes, long num_in_features, long num_out_features) {
    name_ = "Linear";
    num_nodes_ = num_nodes;
    num_in_features_ = num_in_features;
    num_out_features_ = num_out_features;
    memory_usage_ = 2 * num_nodes * num_in_features + num_nodes * num_out_features;
    memory_usage_ = memory_usage_ + 2 * num_in_features * num_out_features + num_out_features + num_nodes;
    a_ = 2 * num_in_features + num_out_features + 1;
    b_ = 2 * num_in_features * num_out_features + num_out_features;
}

FeatureAggregationMemoryModel::FeatureAggregationMemoryModel(long num_nodes, long num_features, long num_edges) {
    name_ = "Feature aggregation";
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    num_edges_ = num_edges;
    // 2 * E + (C + 1) + 2 * C * I + C
    memory_usage_ = 2 * num_edges + num_nodes + 1;
    memory_usage_ = memory_usage_ + 2 * num_nodes * num_features + num_nodes;
    a_ = 1 + 2 * num_features + 1;
    b_ = 2 * num_edges + 1;
}

FeatureAggregationScaledEMemoryModel::FeatureAggregationScaledEMemoryModel(long num_nodes, long num_features, long num_edges) {
    name_ = "Feature aggregation";
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    num_edges_ = num_edges;
    // 2 * E + (C + 1) + 2 * C * I + C
    // replace E with E / (N / C), divide E by number of chunks
    memory_usage_ = 2 * num_edges + num_nodes + 1;
    memory_usage_ = memory_usage_ + 2 * num_nodes * num_features + num_nodes;
    a_ = 2 * (num_edges / num_nodes) + + 2 * num_features + 2;
    b_ = 1;
}

AddMemoryModel::AddMemoryModel(long num_nodes, long num_features) {
    name_ = "Add";
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    memory_usage_ = 2 * num_nodes * num_features;
    a_ = 2 * num_features;
    b_ = 0;
}

// COMPOSITION LAYERS

SAGEConvolutionMemoryModel::SAGEConvolutionMemoryModel(long num_nodes, long num_in_features, long num_out_features, long num_edges) {
    name_ = "SAGE-Convolution";

    layers_ = {new LinearMemoryModel(num_nodes, num_in_features, num_out_features), // self
//               new FeatureAggregationMemoryModel(num_nodes, num_in_features, num_edges),
               new FeatureAggregationScaledEMemoryModel(num_nodes, num_in_features, num_edges),
               new LinearMemoryModel(num_nodes, num_in_features, num_out_features), // neighbourhood
               new AddMemoryModel(num_nodes, num_out_features), // self + neighbourhood
               new AddMemoryModel(num_nodes, num_in_features)}; // self + neighbourhood gradients
}

// GNN MODEL

GraphSAGEMemoryModel::GraphSAGEMemoryModel(long num_layers, long num_nodes, long num_edges,
                                           long num_features, long num_hidden_channels, long num_classes) {
    name_ = "GraphSAGE";

    for (long i = 0; i < num_layers; ++i) {
        if (i == 0) {
            layers_.push_back(new DropoutMemoryModel(num_nodes, num_features));
            layers_.push_back(new SAGEConvolutionMemoryModel(num_nodes, num_features, num_hidden_channels, num_edges));
            layers_.push_back(new ReluMemoryModel(num_nodes, num_hidden_channels));
        } else if (i == num_layers - 1) {
            layers_.push_back(new DropoutMemoryModel(num_nodes, num_hidden_channels));
            layers_.push_back(new SAGEConvolutionMemoryModel(num_nodes, num_hidden_channels, num_classes, num_edges));
            layers_.push_back(new LogSoftmaxMemoryModel(num_nodes, num_classes));
        } else {
            layers_.push_back(new DropoutMemoryModel(num_nodes, num_hidden_channels));
            layers_.push_back(new SAGEConvolutionMemoryModel(num_nodes, num_hidden_channels, num_hidden_channels, num_edges));
            layers_.push_back(new ReluMemoryModel(num_nodes, num_hidden_channels));
        }
    }
}
