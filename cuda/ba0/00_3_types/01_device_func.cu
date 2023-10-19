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

__device__
void fill_id_device(int *workbuf) {
    workbuf[threadIdx.x] = threadIdx.x;
}

__global__
void fill_id(int *workbuf) {
    fill_id_device(workbuf);
}

int main() {
    int hostResult[THREAD_NUM];

    int *deviceWorkbuf;
    cudaMalloc((void **)&deviceWorkbuf, THREAD_NUM*sizeof(int));

    fill_id<<<BLOCK_NUM, THREAD_NUM>>>(deviceWorkbuf);

    cudaMemcpy(hostResult, deviceWorkbuf, THREAD_NUM*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<THREAD_NUM; ++i) {
        printf("%d\n", hostResult[i]);
    }

    cudaFree(deviceWorkbuf);

    return 0;
}