// Copyright 2021 Marcel Wagenländer

#ifndef ALZHEIMER_MEMORY_MODEL_H
#define ALZHEIMER_MEMORY_MODEL_H

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
public:
    virtual ~MemoryModel();
    virtual long get_memory_usage() = 0;
    virtual std::vector<long> get_chunk_size_coefficients() = 0;
};

class DropoutMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    DropoutMemoryModel(long num_nodes, long num_features);
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class ReluMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    ReluMemoryModel(long num_nodes, long num_features);
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class LogSoftmaxMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    LogSoftmaxMemoryModel(long num_nodes, long num_features);
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class LinearMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_in_features_;
    long num_out_features_;

public:
    LinearMemoryModel(long num_nodes, long num_in_features, long num_out_features);
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class FeatureAggregationMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;
    long num_edges_;

public:
    FeatureAggregationMemoryModel(long num_nodes, long num_features, long num_edges);
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class AddMemoryModel : public MemoryModel {
private:
    long num_nodes_;
    long num_features_;

public:
    AddMemoryModel(long num_nodes, long num_features);
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class SAGEConvolutionMemoryModel : public MemoryModel {
private:
    std::vector<MemoryModel *> layers_;

public:
    SAGEConvolutionMemoryModel(long num_nodes, long num_in_features, long num_out_features, long num_edges);
    ~SAGEConvolutionMemoryModel() override;
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
};

class GraphSAGEMemoryModel : public MemoryModel {
private:
    std::vector<MemoryModel *> layers_;

public:
    GraphSAGEMemoryModel(long num_layers, long num_nodes, long num_edges,
                         long num_features, long num_hidden_channels, long num_classes);
    ~GraphSAGEMemoryModel() override;
    long get_memory_usage() override;
    std::vector<long> get_chunk_size_coefficients() override;
    long get_max_chunk_size(long max_num_elements);
};


#endif//ALZHEIMER_MEMORY_MODEL_H
