#include <stdio.h>

 

template<typename Func>
__global__ void kernel(Func f){
    printf("a+Tid=%d\n",f(threadIdx.x));
}

int main(){
    int a = 1000;
    auto f= [=] __host__ __device__ (int tid)->int {return a+tid;};
    kernel<<<1,16>>>(f);
    return 0;
}