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

#define MM_ALLOC malloc
#define MM_FREE free

#define PRLOC printf("%s(), line:%d\n", __func__, __LINE__);

/*
	Assert++
 */
//NOTE: Prepare caller_file & caller_line before use this
#define CALLER_ASSERT3(a) if(a){;}else{printf(" CALLER asserted: in %s(), line %ld\n",caller_file,caller_line);BASIC_ASSERT(0);}
#define CALLER_PARA3 , const char * caller_file, unsigned long caller_line
#define CALLER_ARG3 ,__FUNCTION__,__LINE__

#define DUMPNC(a) printf(#a " = %c\n", a)
#define DUMPNS(a) printf(#a " = %s\n", a)
#define DUMPNSPP(a) printf(#a " = %s\n", a.c_str())
#define DUMPND(a) printf(#a " = %2d\n", (int)(a))
#define DUMPNU(a) printf(#a " = %2u\n", (unsigned int)a)
#define DUMPNX(a) printf(#a " = 0x%X\n", (unsigned int)(a))
#define DUMPNP(a) printf(#a " = %p\n", (void *)(a))
#define DUMPNA(a) printf(#a " = 0x%08X\n", POINTER_TO_U32(a))
#define DUMPNF(a) printf(#a " = %f\n", a)

#define DUMPC(a) printf(#a " = %c, ", a)
#define DUMPS(a) printf(#a " = %s, ", a)
#define DUMPSPP(a) printf(#a " = %s, ", a.c_str())
#define DUMPD(a) printf(#a " = %2d, ", (int)(a))
#define DUMPU(a) printf(#a " = %2u, ", (unsigned int)a)
#define DUMPX(a) printf(#a " = 0x%X, ", (unsigned int)(a))
#define DUMPP(a) printf(#a " = %p, ", (void *)(a))
#define DUMPA(a) printf(#a " = 0x%08X, ", POINTER_TO_U32(a))
#define DUMPF(a) printf(#a " = %f, ", a)

#ifdef __cplusplus
};//extern "C"
#endif

#endif//__BASIC_H_INCLUDED__