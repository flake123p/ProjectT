#ifndef __BASIC_H_INCLUDED__
#define __BASIC_H_INCLUDED__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef byte
typedef unsigned char byte;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define USE_MY_ASSERT (1)
#if USE_MY_ASSERT
#define BASIC_ASSERT(a) if(a){;}else{printf("[FAILED] Assertion failed: in %s(), line %d\n",__FUNCTION__,__LINE__);exit(1);}
#else
#include <assert.h>
#define BASIC_ASSERT assert
#endif

#ifdef __cplusplus
};//extern "C"
#endif

#endif//__BASIC_H_INCLUDED__