#pragma once

//
// https://cplusplus.com/reference/cstdlib/rand/?kw=rand
//
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "_basic.h"
#include "_math.h"

inline void RandSeedInit() {
    srand(time(NULL));
};

template<typename T>
T RandFloat0to1() {
    /*
        rand(): Return an integer value between 0 and RAND_MAX.
    */
    return (T)rand() / (T)RAND_MAX;
}

inline int RandInRange(int start, int end) {
    BASIC_ASSERT(end >= start);
    int width = end - start;
    int ret = rand() % width;
    return ret + start;
};

struct Multinomial {
    float *accu = nullptr;
    size_t accuLen;

    // template<size_t N>
    // Multinomial(float (&_arr)[N]) {
    //     Multinomial(_arr, N);
    // };
    Multinomial(float *arr, size_t len) : accuLen(len) {
        Clear();
        accu = (float *)malloc(sizeof(float)*len);
        // printf("len %lu\n", len);
        // printf("accuLen %lu\n", accuLen);
        float sum = SumOfArray(arr, len);
        if (sum == 0) {
            BASIC_ASSERT(0);
        }
        for (size_t i = 0; i < len; i++) {
            accu[i] = arr[i] / sum;
        }
        for (size_t i = 1; i < len; i++) {
            accu[i] += accu[i-1];
        }
        // for (size_t i = 0; i < len; i++) {
        //     printf("%f\n", accu[i]);
        // }
    };
    ~Multinomial() {
        Clear();
    };
    void Clear() {
        if (accu != nullptr) {
            free(accu);
            accu = nullptr;
        }
    }
    size_t Roll() {
        // printf("accuLen %lu\n", this->accuLen);
        BASIC_ASSERT(this->accuLen != 0);
        size_t ret = 0;
        float rand = RandFloat0to1<float>();
        for (size_t i = 0; i < accuLen; i++) {
            if (accu[i] == 0) {
                ret++;
                continue;
            }
            if (i != 0) {
                if (accu[i] == accu[i-1]) {
                    ret++;
                    continue;
                }
            }
            if (rand < accu[i]) {
                break;
            }
            ret++;
        };
        if (ret == accuLen) {
            ret--;
        } else if (ret > accuLen) {
            BASIC_ASSERT(0);
        }
        return ret;
    }
};