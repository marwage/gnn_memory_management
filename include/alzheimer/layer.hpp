// Copyright 2020 Marcel Wagenländer


class Layer {
    public:
        matrix<float> forward(matrix<float> in);
        matrix<float> backward(matrix<float> gradients);
};

