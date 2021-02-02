// Copyright 2021 Marcel Wagenländer

#ifndef ALZHEIMER_MEMORY_MODEL_H
#define ALZHEIMER_MEMORY_MODEL_H

#include <string>
#include <vector>

class ChunkSizeEquation {
    // a * chunk_size + b = max_num_elements_
protected:
    long a_;
    long b_;
    long max_num_elements_;

public:
    ChunkSizeEquation(long available_memory);
    ChunkSizeEquation(long available_memory, long a, long b);
    void set_coefficients(long a, long b);
    void add(long a, long b);
    long get_chunk_size();
};

class MemoryModel {
protected:
    std::string name_;
    long memory_usage_;
    long a_;
    long b_;

public:
    virtual ~MemoryModel();
    std::string get_name();
    virtual long get_memory_usage();
    virtual std::vector<long> get_chunk_size_coefficients();
    virtual long get_max_chunk_size(long max_num_elements);
};

struct ChunkSizeLayer {
    MemoryModel *layer;
    long chunk_size;
};

struct MemoryUsageLayer {
    MemoryModel *layer;
    long memory_usage;
};

class CompositionMemoryModel : public MemoryModel {
protected:
    std::vector<MemoryModel *> layers_;

public:
    ~CompositionMemoryModel();
    long get_memory_usage() override;
    MemoryUsageLayer get_memory_usage_layer();
    std::vector<long> get_chunk_size_coefficients() override;
    long get_max_chunk_size(long max_num_elements) override;
    ChunkSizeLayer get_max_chunk_size_layer(long max_num_elements);
};

class DropoutMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    DropoutMemoryModel(long num_nodes, long num_features);
};

class ReluMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    ReluMemoryModel(long num_nodes, long num_features);
};

class LogSoftmaxMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    LogSoftmaxMemoryModel(long num_nodes, long num_features);
};

class LinearMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_in_features_;
    long num_out_features_;

public:
    LinearMemoryModel(long num_nodes, long num_in_features, long num_out_features);
};

class FeatureAggregationMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;
    long num_edges_;

public:
    FeatureAggregationMemoryModel(long num_nodes, long num_features, long num_edges);
};

class FeatureAggregationScaledEMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;
    long num_edges_;

public:
    FeatureAggregationScaledEMemoryModel(long num_nodes, long num_features, long num_edges);
};

class AddMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    AddMemoryModel(long num_nodes, long num_features);
};

class SAGEConvolutionMemoryModel : public CompositionMemoryModel {
public:
    SAGEConvolutionMemoryModel(long num_nodes, long num_in_features, long num_out_features, long num_edges);
};

class GraphSAGEMemoryModel : public CompositionMemoryModel {
public:
    GraphSAGEMemoryModel(long num_layers, long num_nodes, long num_edges,
                         long num_features, long num_hidden_channels, long num_classes);
};

#endif//ALZHEIMER_MEMORY_MODEL_H
