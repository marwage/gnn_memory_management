// Copyright 2021 Marcel Wagenländer

// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void transposeNoBankConflicts(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;// transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void transpose(float *mat_T, float *mat, long num_rows, long num_columns) {
    dim3 dimGrid(num_rows / TILE_DIM, num_columns / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(mat_T, mat);
}