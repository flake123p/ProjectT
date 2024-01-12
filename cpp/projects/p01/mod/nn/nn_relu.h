#pragma once

#include "basic.h"

template<typename T>
void nn_relu(T *dst, T *src, size_t num) {
    for (size_t i = 0; i < num; i++) {
        if (src[i] < 0)
            dst[i] = 0;
        else
            dst[i] = src[i];
    }
}
