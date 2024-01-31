#pragma once

#include <cmath>

template<typename T, size_t N>
T SumOfArray( T (&arr)[N]) {
    T ret = (T)0;
    for (decltype(N) i = 0; i < N; i++) {
        ret += arr[i];
    }
    return ret;
}

template<typename T>
T SumOfArray( T *arr, size_t N) {
    T ret = (T)0;
    for (decltype(N) i = 0; i < N; i++) {
        ret += arr[i];
    }
    return ret;
}

template<typename T, typename Out_T>
void SumAndMeanOfArray(T *arr, size_t N, Out_T &sum, Out_T &mean) {
    Out_T sum_ = (Out_T)0;
    for (decltype(N) i = 0; i < N; i++) {
        sum_ += (Out_T)arr[i];
    }
    sum = sum_;
    mean = sum_ / N;
}
