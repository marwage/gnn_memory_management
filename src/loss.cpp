// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"

NLLLoss::NLLLoss(long num_nodes, long num_features) {
    num_nodes_ = num_nodes;
    gradients_.set(num_nodes, num_features, false);
}

float NLLLoss::forward(Matrix<float> *x, Matrix<int> *labels) {
    is_row_major_ = x->is_row_major_;
    loss_ = 0.0;

    for (int i = 0; i < x->num_rows_; ++i) {
        if (is_row_major_) {
            loss_ = loss_ + x->values_[i * x->num_columns_ + labels->values_[i]];
        } else {
            loss_ = loss_ + x->values_[labels->values_[i] * x->num_rows_ + i];
        }
    }
    loss_ = loss_ / (float) x->num_rows_;
    loss_ = -loss_;

    labels_ = labels;

    return static_cast<float>(loss_);
}

Matrix<float> *NLLLoss::backward() {
    gradients_.set_values(0.0);

    for (int i = 0; i < labels_->num_rows_; ++i) {
        if (is_row_major_) {
            gradients_.values_[i * gradients_.num_columns_ + labels_->values_[i]] = -1.0 / num_nodes_;
        } else {
            gradients_.values_[labels_->values_[i] * gradients_.num_rows_ + i] = -1.0 / num_nodes_;
        }
    }

    gradients_.is_row_major_ = is_row_major_;

    return &gradients_;
}

// CHUKING --- CHUNKING --- CHUNKING

NLLLossChunking::NLLLossChunking(long num_nodes, long num_features, long chunk_size) {
    num_nodes_ = num_nodes;
    chunk_size_ = chunk_size;

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        gradients_.at(i).set(current_chunk_size, num_features, false);
    }
}

float NLLLossChunking::forward(std::vector<Matrix<float>> *x, Matrix<int> *labels) {
    is_row_major_ = x->at(0).is_row_major_;
    loss_ = 0.0;

    for (int i = 0; i < num_chunks_; ++i) {
        for (int j = 0; j < x->at(i).num_rows_; ++j) {
            long row = i * chunk_size_ + j;
            if (is_row_major_) {
                loss_ = loss_ + x->at(i).values_[j * x->at(i).num_columns_ + labels->values_[row]];
            } else {
                loss_ = loss_ + x->at(i).values_[labels->values_[row] * x->at(i).num_rows_ + j];
            }
        }
    }

    loss_ = loss_ / (double) num_nodes_;
    loss_ = -loss_;

    labels_ = labels;

    return static_cast<float>(loss_);
}

std::vector<Matrix<float>> *NLLLossChunking::backward() {
    for (int i = 0; i < num_chunks_; ++i) {
        gradients_.at(i).set_values(0.0);
    }

    for (int i = 0; i < num_chunks_; ++i) {
        for (int j = 0; j < gradients_.at(i).num_rows_; ++j) {
            long row = i * chunk_size_ + j;
            if (is_row_major_) {
                gradients_.at(i).values_[j * gradients_.at(i).num_columns_ + labels_->values_[row]] = -1.0 / num_nodes_;
            } else {
                gradients_.at(i).values_[labels_->values_[row] * gradients_.at(i).num_rows_ + j] = -1.0 / num_nodes_;
            }
        }
        gradients_.at(i).is_row_major_ = is_row_major_;
    }

    return &gradients_;
}
