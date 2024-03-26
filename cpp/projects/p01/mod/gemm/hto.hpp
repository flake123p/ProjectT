#pragma once

#include "hto00.hpp"
#include "hto01.hpp"

template<typename Scalar_T, typename HTO_Func_T>
void HtoRowMajor( int m, int n, int k, Scalar_T *a, int lda, 
                                    Scalar_T *b, int ldb,
                                    Scalar_T *c, int ldc )
{
    HTO_Func_T<Scalar_T>(n, m, k, b, ldb, a, lda, c, ldc);
}

template<typename HTO_Func_T>
void HtoRowMajorF32(int m, int n, int k, float *a, float *b, float *c)
{
    HtoRowMajor<float, HTO_Func_T>(n, m, k, b, n, a, k, c, m);
}
