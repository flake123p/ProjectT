#pragma once

#ifndef HPC_API
#define HPC_API
#endif

typedef struct {
    int a;
    int b;
    int c;
    int d;
} Data_t;

class Data2 {
public:
    int a;
    int b;
    int c;
    int d;
    HPC_API void run() {
        d = a * b + c;
    }
};
