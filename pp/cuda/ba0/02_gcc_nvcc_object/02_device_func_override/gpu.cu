#include <stdio.h>
#include "data.h"

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
void myFmadd(Data_t *db) 
{
    db[threadIdx.x].d = db[threadIdx.x].a * db[threadIdx.x].b + db[threadIdx.x].c;
}

void gpu(Data_t *db, int db_num) 
{
    Data_t *deviceWorkbuf;
    cudaMalloc((void **)&deviceWorkbuf, db_num*sizeof(Data_t));

    cudaMemcpy(deviceWorkbuf, db, db_num*sizeof(Data_t), cudaMemcpyHostToDevice);
    myFmadd<<<BLOCK_NUM, db_num>>>(deviceWorkbuf);
    cudaDeviceSynchronize();
    cudaMemcpy(db, deviceWorkbuf, db_num*sizeof(Data_t), cudaMemcpyDeviceToHost);

    cudaFree(deviceWorkbuf);
}

class Data3 : public Data2 {
public:
    __device__ void run() {
        d = a * b + c;
    }
};

__global__
void myRun(Data3 *db) 
{
    db[threadIdx.x].run();
}

void gpu2(Data2 *db, int db_num) 
{
    Data2 *deviceWorkbuf;
    cudaMalloc((void **)&deviceWorkbuf, db_num*sizeof(Data2));

    cudaMemcpy(deviceWorkbuf, db, db_num*sizeof(Data2), cudaMemcpyHostToDevice);
    myRun<<<BLOCK_NUM, db_num>>>(reinterpret_cast<Data3 *>(deviceWorkbuf));
    cudaDeviceSynchronize();
    cudaMemcpy(db, deviceWorkbuf, db_num*sizeof(Data2), cudaMemcpyDeviceToHost);

    cudaFree(deviceWorkbuf);
}