#include <iostream>
#include <thread> // chrono
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define MEM_SIZE_1G   (1*1024*1024*1024)
#define MEM_SIZE_512M (512*1024*1024)

double prof_1x1Gx100_h2d()
{
    printf("%s ...\n", __func__);
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    double t;

    void *host = malloc(MEM_SIZE_1G);
    void *dev;
    cudaStream_t s0;
    cudaStream_t s1;


    checkCudaErrors(cudaMalloc((void **)&dev, MEM_SIZE_1G));
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));

    {
        start = std::chrono::steady_clock::now();
    }
    for (int i = 0; i < 100; i++) {
        checkCudaErrors(cudaMemcpyAsync(dev, host, MEM_SIZE_1G, cudaMemcpyHostToDevice, s0));
        checkCudaErrors(cudaStreamSynchronize(s0));
    }
    {
        end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        t = elapsed_seconds.count(); // t number of seconds, represented as a `double`
    }

    checkCudaErrors(cudaFree(dev));
    free(host);
    return t;
}

double prof_1x1Gx100_d2h()
{
    printf("%s ...\n", __func__);
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    double t;

    void *host = malloc(MEM_SIZE_1G);
    void *dev;
    cudaStream_t s0;
    cudaStream_t s1;


    checkCudaErrors(cudaMalloc((void **)&dev, MEM_SIZE_1G));
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));

    {
        start = std::chrono::steady_clock::now();
    }
    for (int i = 0; i < 100; i++) {
        checkCudaErrors(cudaMemcpyAsync(host, dev, MEM_SIZE_1G, cudaMemcpyDeviceToHost, s0));
        checkCudaErrors(cudaStreamSynchronize(s0));
    }
    {
        end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        t = elapsed_seconds.count(); // t number of seconds, represented as a `double`
    }

    checkCudaErrors(cudaFree(dev));
    free(host);
    return t;
}

double prof_2x512Mx100()
{
    printf("%s ...\n", __func__);
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    double t;

    void *host1 = malloc(MEM_SIZE_512M);
    void *dev1;
    void *host2 = malloc(MEM_SIZE_512M);
    void *dev2;
    cudaStream_t s0;
    cudaStream_t s1;


    checkCudaErrors(cudaMalloc((void **)&dev1, MEM_SIZE_512M));
    checkCudaErrors(cudaMalloc((void **)&dev2, MEM_SIZE_512M));
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));

    {
        start = std::chrono::steady_clock::now();
    }
    for (int i = 0; i < 100; i++) {
        checkCudaErrors(cudaMemcpyAsync(dev1, host1, MEM_SIZE_512M, cudaMemcpyHostToDevice, s0));
        checkCudaErrors(cudaMemcpyAsync(host2, dev2, MEM_SIZE_512M, cudaMemcpyDeviceToHost, s1));
        checkCudaErrors(cudaStreamSynchronize(s0));
        checkCudaErrors(cudaStreamSynchronize(s1));
    }
    {
        end = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_seconds = end - start;
        t = elapsed_seconds.count(); // t number of seconds, represented as a `double`
    }

    checkCudaErrors(cudaFree(dev1));
    checkCudaErrors(cudaFree(dev2));
    free(host1);
    free(host2);
    return t;
}

int main() 
{
    printf("cudaGetDeviceCount\n");
    int devCount;
    cudaError_t rc = cudaGetDeviceCount(&devCount);
    printf("RC = %d, devCount = %d\n", (int)rc, devCount);

    for (int devIdx = 0; devIdx < devCount; devIdx ++) 
    {
        printf("cudaSetDevice: %d\n", devIdx);
        rc = cudaSetDevice(devIdx);
        printf("RC = %d\n", rc);

        printf("cudaGetDeviceProperties\n");
        struct cudaDeviceProp pp;
        rc = cudaGetDeviceProperties(&pp, devIdx);
        printf("RC = %d\n", rc);
        printf("name = %s\n", pp.name);

        if (1)
        {
            double t = prof_1x1Gx100_h2d();
            printf("t = %f(%.2f) seconds, GBs = %f(%.2f)\n", t, t, 100.0/t, 100.0/t);
        }
        if (1)
        {
            double t = prof_1x1Gx100_d2h();
            printf("t = %f(%.2f) seconds, GBs = %f(%.2f)\n", t, t, 100.0/t, 100.0/t);
        }
        if (1)
        {
            double t = prof_2x512Mx100();
            printf("t = %f(%.2f) seconds, GBs = %f(%.2f)\n", t, t, 100.0/t, 100.0/t);
        }
    }

    return 0;
}