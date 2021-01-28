// Copyright 2021 Marcel Wagenländer

#include "memory_model.hpp"

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

MemoryModel::~MemoryModel() {}

// DROPOUT

DropoutMemoryModel::DropoutMemoryModel(long num_nodes, long num_features) {
    num_nodes_ = num_nodes;
    num_features_ = num_features;
}

long DropoutMemoryModel::get_memory_usage() {
    return 2 * num_nodes_ * num_features_;
}

std::vector<long> DropoutMemoryModel::get_chunk_size_coefficients() {
    return {2 * num_features_, 0};
}

// RELU

ReluMemoryModel::ReluMemoryModel(long num_nodes, long num_features) {
    num_nodes_ = num_nodes;
    num_features_ = num_features;
}

long ReluMemoryModel::get_memory_usage() {
    return 4 * num_nodes_ * num_features_;
}

std::vector<long> ReluMemoryModel::get_chunk_size_coefficients() {
    return {4 * num_features_, 0};
}

// LOG-SOFTMAX

LogSoftmaxMemoryModel::LogSoftmaxMemoryModel(long num_nodes, long num_features) {
    num_nodes_ = num_nodes;
    num_features_ = num_features;
}

long LogSoftmaxMemoryModel::get_memory_usage() {
    return 3 * num_nodes_ * num_features_;
}

std::vector<long> LogSoftmaxMemoryModel::get_chunk_size_coefficients() {
    return {3 * num_features_, 0};
}

// LINEAR

LinearMemoryModel::LinearMemoryModel(long num_nodes, long num_in_features, long num_out_features) {
    num_nodes_ = num_nodes;
    num_in_features_ = num_in_features;
    num_out_features_ = num_out_features;
}

long LinearMemoryModel::get_memory_usage() {
    // 2 * N * I + N * O + 2 * I * O + O + N
    long max_memory = 2 * num_nodes_ * num_in_features_ + num_nodes_ * num_out_features_;
    max_memory = max_memory + 2 * num_in_features_ * num_out_features_ + num_out_features_ + num_nodes_;
    return max_memory;
}

std::vector<long> LinearMemoryModel::get_chunk_size_coefficients() {
    // 2 * C * I + C * O + 2 * I * O + O + C
    long a = 2 * num_in_features_ + num_out_features_ + 1;
    long b = 2 * num_in_features_ * num_out_features_ + num_out_features_;
    return {a, b};
}

// FEATURE AGGREGATION

FeatureAggregationMemoryModel::FeatureAggregationMemoryModel(long num_nodes, long num_features, long num_edges) {
    num_nodes_ = num_nodes;
    num_features_ = num_features;
    num_edges_ = num_edges;
}

long FeatureAggregationMemoryModel::get_memory_usage() {
    // 2 * N * I + 2 * E + (N + 1) + N
    long max_memory = 2 * num_nodes_ * num_features_ + 2 * num_edges_;
    max_memory = max_memory + num_nodes_ + 1 + num_nodes_;
    return max_memory;
}

std::vector<long> FeatureAggregationMemoryModel::get_chunk_size_coefficients() {
    // 2 * E + (C + 1) + 2 * C * I + C
    long a = 1 + 2 * num_features_ + 1;
    long b = 2 * num_edges_ + 1;
    return {a, b};
}

// ADD

AddMemoryModel::AddMemoryModel(long num_nodes, long num_features) {
    num_nodes_ = num_nodes;
    num_features_ = num_features;
}

long AddMemoryModel::get_memory_usage() {
    return 3 * num_nodes_ * num_features_;
}

std::vector<long> AddMemoryModel::get_chunk_size_coefficients() {
    return {3 * num_features_, 0};
}


// SAGE CONVOLUTION

SAGEConvolutionMemoryModel::SAGEConvolutionMemoryModel(long num_nodes, long num_in_features, long num_out_features, long num_edges) {
    layers_ = {new LinearMemoryModel(num_nodes, num_in_features, num_out_features), // self
               new FeatureAggregationMemoryModel(num_nodes, num_in_features, num_edges),
               new LinearMemoryModel(num_nodes, num_in_features, num_out_features), // neighbourhood
               new AddMemoryModel(num_nodes, num_out_features), // self + neighbourhood
               new AddMemoryModel(num_nodes, num_in_features)}; // self + neighbourhood gradients
}

SAGEConvolutionMemoryModel::~SAGEConvolutionMemoryModel() {
    for (MemoryModel *layer : layers_) {
        delete layer;
    }
}

long SAGEConvolutionMemoryModel::get_memory_usage() {
    long max_memory = 0;
    for (MemoryModel *layer : layers_) {
        max_memory = max_memory + layer->get_memory_usage();
    }
    return max_memory;
}

std::vector<long> SAGEConvolutionMemoryModel::get_chunk_size_coefficients() {
    long a = 0;
    long b = 0;
    for (MemoryModel *layer : layers_) {
        std::vector<long> vars = layer->get_chunk_size_coefficients();
        a = a + vars.at(0);
        b = b + vars.at(1);
    }
    return {a, b};
}

// GraphSAGE

GraphSAGEMemoryModel::GraphSAGEMemoryModel(long num_layers, long num_nodes, long num_edges,
                                           long num_features, long num_hidden_channels, long num_classes) {
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

GraphSAGEMemoryModel::~GraphSAGEMemoryModel() {
    for (MemoryModel *layer : layers_) {
        delete layer;
    }
}

long GraphSAGEMemoryModel::get_memory_usage() {
    long max_memory = 0;
    for (MemoryModel *layer : layers_) {
        long layer_max_memory = layer->get_memory_usage();
        max_memory = max_memory + layer_max_memory;
        if (max_memory < 0) throw "Memory smaller 0";
    }

    return max_memory;
}

std::vector<long> GraphSAGEMemoryModel::get_chunk_size_coefficients() {
    long a = 0;
    long b = 0;
    for (MemoryModel *layer : layers_) {
        std::vector<long> vars = layer->get_chunk_size_coefficients();
        a = a + vars.at(0);
        b = b + vars.at(1);
    }
    return {a, b};
}

long GraphSAGEMemoryModel::get_max_chunk_size(long max_num_elements) {
    std::vector<long> coefficients = GraphSAGEMemoryModel::get_chunk_size_coefficients();

    ChunkSizeEquation chunk_size_eq(max_num_elements, coefficients.at(0), coefficients.at(1));

    return chunk_size_eq.get_chunk_size();
}
