// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "tensors.hpp"
#include "linear.hpp"


class SageLinear {
 private:
    int num_in_features_;
    int num_out_features_;
    Linear linear_self_;
    Linear linear_neigh_;
 public:
    SageLinear(int in_features, int out_features);
    matrix<float> forward(matrix<float> features, matrix<float> aggr);
};

#endif

