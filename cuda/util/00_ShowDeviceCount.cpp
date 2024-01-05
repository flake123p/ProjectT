#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

int main() 
{
    {
        printf("cudaGetDeviceCount\n");
        int devCount;
        cudaError_t rc = cudaGetDeviceCount(&devCount);
        printf("RC = %d, devCount = %d\n", (int)rc, devCount);
    }

    return 0;
}