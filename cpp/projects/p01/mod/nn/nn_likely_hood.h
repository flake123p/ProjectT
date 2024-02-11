#pragma once

#include "basic.h"
#include <cmath>

template<typename T>
T nn_likely_hood(T *src1, T *src2, size_t num) {
    T sum = (T)0;

    for (size_t i = 0; i < num; i++) {
        sum += pow(src1[i], src2[i]) * pow(((T)1-src1[i]), ((T)1-src2[i]));
    }

    return sum;
}
