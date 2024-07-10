#pragma once

#ifndef HPC_API
#define HPC_API
#endif

class EUnit {
public:
    int a[4];
    HPC_API void run() {
        a[3] = a[0] * a[1] + a[2];
    }
};
void runThreads(EUnit *units, int num);
void runGpu(EUnit *units, int num);