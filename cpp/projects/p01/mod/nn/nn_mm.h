#pragma once

#include "basic.h"

template<typename T>
inline T mm_result(T &sum, T* C, T alpha, T beta) {
    if (C == nullptr) {
        return sum * alpha;
    } else {
        return (sum * alpha) + (*C * beta);
    }
}

template<typename T>
int nn_Matmul_RowMajor(T *A, T *B, T *C, T *D, int m, int k, int n, int A_T, int B_T, T alpha = (T)1, T beta = (T)0)
{
    printf(">>> %s(), [%u,%u - %u,%u][AT=%u, BT=%u]\n", __func__, m, k, k, n, A_T, B_T);

    const T* f_A = (const T*)A;
    const T* f_B = (const T*)B;
    T* f_C = (T*)C;
    T* f_D = (T*)D;
    const T f_alpha = (const T)alpha;
    const T f_beta = (const T)beta;
    //uint32_t i, j, y, idc, idk;

    // SWC_PRINT("MAT_size %d %d %d\n",m,k,n);
    // for( i = 0; i < m*k; i++){
    //     SWC_PRINT("A[%d] %f\n",i,f_A[i]);
    // }
    // for( i = 0; i < n*k; i++){
    //     SWC_PRINT("B[%d] %f\n",i,f_B[i]);
    // }
    {
        // to row major
        // f_A = (const T*)B;
        // f_B = (const T*)A;
        // uint32_t m_ori = m;
        // uint32_t n_ori = n;
        // m = n_ori;
        // n = m_ori;
        // uint32_t A_T_ori = A_T;
        // uint32_t B_T_ori = B_T;
        // A_T = B_T_ori;
        // B_T = A_T_ori;

        T* dst = f_D;
        T* srcC = f_C;
        if(A_T == 0 && B_T == 0) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idm*k) + idk] * f_B[(idk*n) + idn];
                    }
                    *dst = mm_result(sum, srcC, f_alpha, f_beta);
                    dst++;
                    srcC++;
                }
            }
        }
        else if(A_T == 0 && B_T == 1) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idm*k) + idk] * f_B[(idn*k) + idk];
                    }
                    *dst = mm_result(sum, srcC, f_alpha, f_beta);
                    dst++;
                    srcC++;
                }
            }
        }
        else if(A_T == 1 && B_T == 0) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idk*m) + idm] * f_B[(idk*n) + idn];
                    }
                    *dst = mm_result(sum, srcC, f_alpha, f_beta);
                    dst++;
                    srcC++;
                }
            }
        }
        else if(A_T == 1 && B_T == 1) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idk*m) + idm] * f_B[(idn*k) + idk];
                    }
                    *dst = mm_result(sum, srcC, f_alpha, f_beta);
                    dst++;
                    srcC++;
                }
            }
        }
    }

    return 0;
}

template<typename T>
int nn_MatmulLt_RowMajor(T *A, T *B, T *dst, int m, int k, int n, int A_T, int B_T)
{
    //printf(">>> %s(), [%u,%u - %u,%u][AT=%u, BT=%u]\n", __func__, m, k, k, n, A_T, B_T);
    const T* f_A = (const T*)A;
    const T* f_B = (const T*)B;
    //uint32_t i, j, y, idc, idk;

    // SWC_PRINT("MAT_size %d %d %d\n",m,k,n);
    // for( i = 0; i < m*k; i++){
    //     SWC_PRINT("A[%d] %f\n",i,f_A[i]);
    // }
    // for( i = 0; i < n*k; i++){
    //     SWC_PRINT("B[%d] %f\n",i,f_B[i]);
    // }
    {
        // to row major
        // f_A = (const T*)B;
        // f_B = (const T*)A;
        // uint32_t m_ori = m;
        // uint32_t n_ori = n;
        // m = n_ori;
        // n = m_ori;
        // uint32_t A_T_ori = A_T;
        // uint32_t B_T_ori = B_T;
        // A_T = B_T_ori;
        // B_T = A_T_ori;

        if(A_T == 0 && B_T == 0) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idm*k) + idk] * f_B[(idk*n) + idn];
                    }
                    *dst = sum;
                    dst++;
                }
            }
        }
        else if(A_T == 0 && B_T == 1) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idm*k) + idk] * f_B[(idn*k) + idk];
                    }
                    *dst = sum;
                    dst++;
                }
            }
        }
        else if(A_T == 1 && B_T == 0) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idk*m) + idm] * f_B[(idk*n) + idn];
                    }
                    *dst = sum;
                    dst++;
                }
            }
        }
        else if(A_T == 1 && B_T == 1) {
            for (int idm = 0; idm < m; idm++) {
                for (int idn = 0; idn < n; idn++) {
                    T sum = 0.0;
                    for (int idk = 0; idk < k; idk++) {
                        sum += f_A[(idk*m) + idm] * f_B[(idn*k) + idk];
                    }
                    *dst = sum;
                    dst++;
                }
            }
        }
    }

    return 0;
}