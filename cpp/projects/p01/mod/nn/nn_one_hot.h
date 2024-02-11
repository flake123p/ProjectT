#pragma once

#include "basic.h"

template<typename T_DST, typename T_SRC>
void nn_one_hot_encode(T_DST *dst, T_SRC src, size_t num) {
    printf("src = %d\n", src);
    for (size_t i = 0; i < num; i++) {
        if ((size_t)src == i) {
            dst[i] = (T_DST)1;
        } else {
            dst[i] = (T_DST)0;
        }
    }
}
