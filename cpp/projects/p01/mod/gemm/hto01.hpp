#pragma once

#include "hto_internal.hpp"

// From: https://github.com/flame/how-to-optimize-gemm

/* Create macros so that the matrices are stored in column-major order */

/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

template<typename Scalar_T>
void AddDot( int k, Scalar_T *x, int incx,  Scalar_T *y, Scalar_T *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */
 
  int p;

  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];     
  }
}

/* Routine for computing C = A * B + C */
template<typename Scalar_T>
void hto_mm_01( int m, int n, int k, Scalar_T *a, int lda, 
                                    Scalar_T *b, int ldb,
                                    Scalar_T *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=1 ){        /* Loop over the columns of C */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	      and the jth column of B */

      AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
    }
  }
}