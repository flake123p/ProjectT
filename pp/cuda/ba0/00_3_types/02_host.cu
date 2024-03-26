#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//

// __global__ = cpu    call, run on device
// __device__ = device call, run on device
// __host__   = cpu    call, run on    cpu

#define BLOCK_NUM 1
#define THREAD_NUM 4

__host__
void fill_id(int *workbuf) {
    workbuf[0] = 999;
}

int main() {
    int hostResult[THREAD_NUM] = {0};

    fill_id(hostResult);

    for (int i = 0; i<THREAD_NUM; ++i) {
        printf("%d\n", hostResult[i]);
    }

    return 0;
}