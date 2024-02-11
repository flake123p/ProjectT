#pragma once

#include "basic.h"
#include <cmath> //exp()

template<typename T>
T nn_softmax_op_11(T &src) {
    return (T)pow(1.1, src);
}

template<typename T>
void nn_softmax_11(T *dst, T *src, size_t num) {
    T sum = (T)0;

    for (size_t i = 0; i < num; i++) {
        sum += nn_softmax_op_11(src[i]);
        printf("src[%lu] = %f\n", i, nn_softmax_op_11(src[i]));
    }

    for (size_t i = 0; i < num; i++) {
        printf("[%lu] = %f / %f / %f\n", i, src[i], nn_softmax_op_11(src[i])/sum, sum);
        dst[i] = nn_softmax_op_11(src[i]) / sum;
    }
}