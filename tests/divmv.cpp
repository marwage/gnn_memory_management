// Copyright 2020 Marcel Wagenländer


void check_divmv() {
    float *d_X, *d_y;
    int n = 500;
    int m = 700;

    check_cuda(cudaMalloc((void **) &d_X, n * m * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_y, n * sizeof(float)));

    float X[n * m];
    for (int i = 0; i < n * m; ++i) {
        X[i] = 1.0;
    }
    float y[n];
    for (int i = 0; i < n; ++i) {
        y[i] = (float) (i + 1);
    }

    check_cuda(cudaMemcpy(d_X, X, n * m * sizeof(float),
                cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_y, y, n * sizeof(float),
                cudaMemcpyHostToDevice));

    div_mat_vec(d_X, d_y, n, m);

    check_cuda(cudaMemcpy(X, d_X, n * m * sizeof(float),
                cudaMemcpyDeviceToHost));
    
    matrix<float> X_mat;
    X_mat.rows = n;
    X_mat.columns = m;
    X_mat.values = X;
    print_matrix<float>(X_mat);

    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_y));
}

