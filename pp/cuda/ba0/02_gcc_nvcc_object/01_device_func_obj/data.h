#pragma once

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
    __device__ void run() {
        d = a * b + c;
    }
};
