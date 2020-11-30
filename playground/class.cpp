// Copyright 2020 Marcel Wagenländer


class Linear {
private:
    int num_in_features_;
    int num_out_features_;

public:
    Linear();
    Linear(int in_features, int out_features);
};

Linear::Linear() {
    num_in_features_ = 0;
    num_out_features_ = 0;
}

Linear::Linear(int in_features, int out_features) {
    num_in_features_ = in_features;
    num_out_features_ = out_features;
}

class SageLinear {
private:
    Linear linear;

public:
    SageLinear(int in_features, int out_features);
};

SageLinear::SageLinear(int in_features, int out_features) {
    linear = Linear(in_features, out_features);
}

int main() {
    SageLinear sage(5, 4);
}
