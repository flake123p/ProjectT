
/* Create macros so that the matrices are stored in column-major order */

#ifndef A
#define A(i,j) a[ (j)*lda + (i) ]
#endif
#ifndef B
#define B(i,j) b[ (j)*ldb + (i) ]
#endif
#ifndef C
#define C(i,j) c[ (j)*ldc + (i) ]
#endif

/* Routine for computing C = A * B + C */

template<typename Scalar_T>
void MY_MMult_Opti00( int m, int n, int k, Scalar_T *a, int lda, 
                                    Scalar_T *b, int ldb,
                                    Scalar_T *c, int ldc )
{
  int i, j, p;

  for ( i=0; i<m; i++ ){        /* Loop over the rows of C */
    for ( j=0; j<n; j++ ){        /* Loop over the columns of C */
      for ( p=0; p<k; p++ ){        /* Update C( i,j ) with the inner
				       product of the ith row of A and
				       the jth column of B */
	      C( i,j ) = C( i,j ) +  A( i,p ) * B( p,j );
      }
    }
  }
}