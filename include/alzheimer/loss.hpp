// Copyright 2020 Marcel Wagenländer

#ifndef LOSS_H
#define LOSS_H

#include "tensors.hpp"


float negative_log_likelihood_loss(matrix<float> X, vector<int> y);

#endif
