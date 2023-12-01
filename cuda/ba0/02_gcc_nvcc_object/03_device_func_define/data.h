#pragma once

#ifndef COMMON_API
#define COMMON_API
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
    COMMON_API void run() {
        d = a * b + c;
    }
};
