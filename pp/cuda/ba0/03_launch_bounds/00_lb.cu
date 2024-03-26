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


int main() {

    return 0;
}