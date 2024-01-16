#pragma once

//
// https://cplusplus.com/reference/cstdlib/rand/?kw=rand
//
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "basic.h"

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