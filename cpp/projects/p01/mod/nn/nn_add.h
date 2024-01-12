#pragma once

#include "basic.h"

template<typename T>
void nn_vecAdd(T *dst, T *src1, T *src2, size_t num) {
    for (size_t i = 0; i < num; i++) {
        dst[i] = src1[i] + src2[i];
    }
}
