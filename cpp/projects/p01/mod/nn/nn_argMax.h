#pragma once

#include "basic.h"

template<typename T>
size_t nn_argMax(T *buf, size_t num) {
    size_t idxMax = 0;
    T max = buf[0];
    for (size_t i = 1; i < num; i++) {
        if (buf[i] > max) {
            max = buf[i];
            idxMax = i;
        }
    }
    return idxMax;
}
