#pragma once

/*
    Lib stat is based on Lib basic !!!!
*/
#include "_basic.h"
#include "_lib.h"

//SumOfArray

template<typename T, typename Out_T>
void Stat_SumMeanVarianceStddev(T *arr, size_t N, Out_T &sum, Out_T &mean, Out_T &var, Out_T &sd) {
    SumAndMeanOfArray(arr, N, sum, mean);
    
    Out_T *diff = (Out_T *)malloc(N * sizeof(Out_T));

    Out_T *prob = (Out_T *)malloc(N * sizeof(Out_T));

    for (size_t i = 0; i < N; i++) {
        diff[i] = (Out_T)arr[i] - mean;
    }

    for (size_t i = 0; i < N; i++) {
        prob[i] = (Out_T)1 / N;
    }

    var = 0;
    for (size_t i = 0; i < N; i++) {
        var += diff[i] * diff[i];// * prob[i];
    }
    var /= N;

    sd = sqrt(var);

    free(diff);
    free(prob);
}