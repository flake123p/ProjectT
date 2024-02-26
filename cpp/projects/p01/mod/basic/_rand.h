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
    float *src = nullptr;
    float *maskedSrcVal = nullptr;
    float *accu = nullptr;
    float *selectableMask = nullptr;
    float *output = nullptr;
    size_t accuLen = 0;

    float CumsumSelectable() {
        BASIC_ASSERT(accu != nullptr);
        BASIC_ASSERT(maskedSrcVal != nullptr);
        for (size_t i = 0; i < accuLen; i++) {
            maskedSrcVal[i] = src[i] * selectableMask[i];
        }
        
        float sum = SumOfArray(maskedSrcVal, accuLen);

        //printf("CumsumSelectable, A sum=%f, %f %f %f %f\n", sum, maskedSrcVal[0], maskedSrcVal[1], maskedSrcVal[2], maskedSrcVal[3]);
        if (sum == 0) {
            for (size_t i = 0; i < accuLen; i++) {
                maskedSrcVal[i] = selectableMask[i];
            }
            sum = SumOfArray(maskedSrcVal, accuLen);
            BASIC_ASSERT(sum != 0);
        }
        BASIC_ASSERT(sum != 0);
        for (size_t i = 0; i < accuLen; i++) {
            accu[i] = maskedSrcVal[i] / sum;
            BASIC_ASSERT(accu[i] <= 1.0);
        }
        for (size_t i = 1; i < accuLen; i++) {
            accu[i] += accu[i-1];
            // if (accu[i] > 1.0) {
            //     if (accu[i] < 1.0001) {
            //         accu[i] = 1.0;
            //     }
            //     //printf("i:%lu, accuLen:%lu, accu[i]:%f\n", i, accuLen, accu[i]);
            // }
            // BASIC_ASSERT(accu[i] <= 1.0);
        }
        //printf("CumsumSelectable, B sum=%f, %f %f %f %f\n", sum, accu[0], accu[1], accu[2], accu[3]);
        return sum;
    }

    // template<size_t N>
    // Multinomial(float (&_arr)[N]) {
    //     Multinomial(_arr, N);
    // };
    Multinomial(float *arr, size_t len) : accuLen(len) {
        Clear();
        this->accu = (float *)malloc(sizeof(float)*len);
        this->maskedSrcVal = (float *)malloc(sizeof(float)*len);
        this->src = arr;
        this->selectableMask = (float *)malloc(sizeof(float)*len);
        this->output = (float *)malloc(sizeof(float)*len);
        
        for (size_t i = 0; i < len; i++) {
            this->selectableMask[i] = 1.0f;
        }
        // printf("len %lu\n", len);
        // printf("accuLen %lu\n", accuLen);

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
        CumsumSelectable();
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

    size_t RollWithMask() {
        size_t selected = 0;

        BASIC_ASSERT(accu != nullptr);
        BASIC_ASSERT(maskedSrcVal != nullptr);

        float sum = CumsumSelectable();
        if (sum == 0.0) {
            for (size_t i = 0; i < accuLen; i++) {
                if (src[i] == 0 && selectableMask[i] == 1.0f) {
                    return i;
                }
            }
            return 0;
        }
        
        float rand = RandFloat0to1<float>();
        //printf("r=%f, %f %f %f %f\n", rand, accu[0], accu[1], accu[2], accu[3]);
        for (size_t i = 0; i < accuLen; i++) {
            if (accu[i] == 0) {
                selected++;
                continue;
            }
            if (i != 0) {
                if (accu[i] == accu[i-1]) {
                    selected++;
                    continue;
                }
            }
            if (rand < accu[i]) {
                break;
            }
            selected++;
        };
        if (selected == accuLen) {
            selected--;
        } else if (selected > accuLen) {
            BASIC_ASSERT(0);
        }
        return selected;
    }

    void RollAll() {
        float currIdx = (float)accuLen;
        size_t selected;
        for (size_t i = 0; i < accuLen; i++) {
            this->selectableMask[i] = 1.0f;
        }
        for (size_t i = 0; i < accuLen; i++) {
            selected = RollWithMask();
            
            output[selected] = currIdx;
            currIdx -= 1.0;
            this->selectableMask[selected] = 0.0f;

            //printf("selected = %lu, %f %f %f %f\n", selected, selectableMask[0], selectableMask[1], selectableMask[2], selectableMask[3]);
        }
    }

    void CalcOutput() {
        RollAll();

        float epsilon = 1e-5;

        for (size_t i = 0; i < accuLen; i++) {
            output[i] = (src[i] + epsilon) / output[i];
        }
    }
};