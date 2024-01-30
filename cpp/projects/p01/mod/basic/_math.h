#pragma once

#include <cmath>

template<typename T, size_t N>
T SumOfArray( T (&_arr)[N]) {
    T ret = (T)0;
    for (decltype(N) i = 0; i < N; i++) {
        ret += _arr[i];
    }
    return ret;
}

template<typename T>
T SumOfArray( T *_arr, size_t N) {
    T ret = (T)0;
    for (decltype(N) i = 0; i < N; i++) {
        ret += _arr[i];
    }
    return ret;
}