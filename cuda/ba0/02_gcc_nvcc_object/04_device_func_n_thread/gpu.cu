#include <stdio.h>
#define HPC_API __device__
#include "hpc.h"

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

__global__
void myRun(EUnit *units) 
{
    //printf("%d, %d\n", threadIdx.x, units[threadIdx.x].a[0]);
    units[threadIdx.x].run();
}

void runGpu(EUnit *units, int num)
{
    EUnit *deviceWorkbuf;
    cudaMalloc((void **)&deviceWorkbuf, num*sizeof(*deviceWorkbuf));

    cudaMemcpy(deviceWorkbuf, units, num*sizeof(*deviceWorkbuf), cudaMemcpyHostToDevice);
    myRun<<<BLOCK_NUM, num>>>(deviceWorkbuf);
    cudaDeviceSynchronize();
    cudaMemcpy(units, deviceWorkbuf, num*sizeof(*deviceWorkbuf), cudaMemcpyDeviceToHost);

    cudaFree(deviceWorkbuf);
}