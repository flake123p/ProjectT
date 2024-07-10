
# Flake

## CUDA examples


## MOVE/RVO/NRVO
https://learn.microsoft.com/en-us/previous-versions/ms364057(v=vs.80)?redirectedfrom=MSDN
Move semantics also helps when the compiler cannot use Return Value Optimization (RVO) or Named Return Value Optimization (NRVO). 
In these cases, the compiler calls the move constructor if the type defines it. 
For more information about Named Return Value Optimization, see Named Return Value Optimization in Visual C++ 2005.


# inline / static inline (pytorch)

static inline cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                                        cublasOperation_t transa,
                                                        cublasOperation_t transb,
                                                        int m,
                                                        int n,
                                                        int k,
                                                        const void* alpha, /* host or device pointer */
                                                        const void* A,
                                                        cudaDataType Atype,
                                                        int lda,
                                                        long long int strideA, /* purposely signed */
                                                        const void* B,
                                                        cudaDataType Btype,
                                                        int ldb,
                                                        long long int strideB,
                                                        const void* beta, /* host or device pointer */
                                                        void* C,
                                                        cudaDataType Ctype,
                                                        int ldc,
                                                        long long int strideC,
                                                        int batchCount,
                                                        cudaDataType computeType,
                                                        cublasGemmAlgo_t algo) {
  cublasComputeType_t migratedComputeType;
  cublasStatus_t status;
  status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  return cublasGemmStridedBatchedEx(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    Atype,
                                    lda,
                                    strideA,
                                    B,
                                    Btype,
                                    ldb,
                                    strideB,
                                    beta,
                                    C,
                                    Ctype,
                                    ldc,
                                    strideC,
                                    batchCount,
                                    migratedComputeType,
                                    algo);