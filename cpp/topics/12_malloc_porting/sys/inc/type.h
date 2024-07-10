#ifndef _type_H_INCLUDED_
#define _type_H_INCLUDED_

#include <stdint.h> // flake: may not exist on embedded system

#ifdef _USING_FLOAT_32_

typedef float FLOAT_T;
//typedef double FLOAT_T;
typedef float DOUBLE_T;
//typedef double DOUBLE_T;
typedef int64_t LONG_SINT_T;
typedef uint64_t LONG_UINT_T;
typedef float F32_T;
typedef float F64_T;
#define EMBCV_FLT_EPSILON  FLT_EPSILON
#define EMBCV_DBL_EPSILON  FLT_EPSILON
#define EMBCV_FLT_MAX  FLT_MAX
#define EMBCV_DBL_MAX  FLT_MAX
#define EMBCV_CV_64F  CV_32F

#else //#ifdef _USING_FLOAT_32_

//typedef float FLOAT_T;
typedef double FLOAT_T;
//typedef float DOUBLE_T;
typedef double DOUBLE_T;
typedef int64_t LONG_SINT_T;
typedef uint64_t LONG_UINT_T;
typedef float F32_T;
typedef double F64_T;
#define _RETAIN_F64_OVERLOAD_
#define EMBCV_FLT_EPSILON  FLT_EPSILON
#define EMBCV_DBL_EPSILON  DBL_EPSILON
#define EMBCV_FLT_MAX  FLT_MAX
#define EMBCV_DBL_MAX  DBL_MAX
#define EMBCV_CV_64F  CV_64F

#endif //#ifdef _USING_FLOAT_32_

#ifdef _USING_MEM_BUS_32_
#define POINTER_SIZE_TYPE    uint32_t
#else
#define POINTER_SIZE_TYPE    uint64_t
#endif

#endif//_type_H_INCLUDED_

