


#ifndef _MY_MACROFUNCS_H_INCLUDED_







#define FOREACH_I(max) for(unsigned int i=0;i<max;i++)
#define FOR_I(max)     FOREACH_I(max)
#define FOREACH_J(max) for(unsigned int j=0;j<max;j++)
#define FOR_J(max)     FOREACH_J(max)


#include <stdint.h>
// uintptr_t is defined in C++11 and later standards.
#define POINTER_TO_INT(ptr) ((uintptr_t)ptr)
#define POINTER_TO_U32(ptr) ((u32)POINTER_TO_INT(ptr))

#define DUMPNC(a) printf(#a " = %c\n", a)
#define DUMPNS(a) printf(#a " = %s\n", a)
#define DUMPNSPP(a) printf(#a " = %s\n", a.c_str())
#define DUMPND(a) printf(#a " = %2d\n", (int)(a))
#define DUMPNU(a) printf(#a " = %2u\n", (unsigned int)a)
#define DUMPNX(a) printf(#a " = 0x%X\n", (unsigned int)(a))
#define DUMPNP(a) printf(#a " = %p\n", (void *)(a))
#define DUMPNA(a) printf(#a " = 0x%08X\n", POINTER_TO_U32(a))
#define DUMPNF(a) printf(#a " = %f\n", a)

#define DUMPNX8(a) printf(#a " = 0x%02X\n", (unsigned int)(a))
#define DUMPNX16(a) printf(#a " = 0x%04X\n", (unsigned int)(a))
#define DUMPNX32(a) printf(#a " = 0x%08X\n", (unsigned int)(a))
#define DUMPNX_LIBU64(a) DUMPNX32((a).hi);DUMPNX32((a).lo);

#define DUMPC(a) printf(#a " = %c, ", a)
#define DUMPS(a) printf(#a " = %s, ", a)
#define DUMPSPP(a) printf(#a " = %s, ", a.c_str())
#define DUMPD(a) printf(#a " = %2d, ", (int)(a))
#define DUMPU(a) printf(#a " = %2u, ", (unsigned int)a)
#define DUMPX(a) printf(#a " = 0x%X, ", (unsigned int)(a))
#define DUMPP(a) printf(#a " = %p, ", (void *)(a))
#define DUMPA(a) printf(#a " = 0x%08X, ", POINTER_TO_U32(a))
#define DUMPF(a) printf(#a " = %f, ", a)

#define DNC(a) DUMPNC(a)
#define DNS(a) DUMPNS(a)
#define DNSPP(a) DUMPNSPP(a)
#define DND(a) DUMPND(a)
#define DNU(a) DUMPNU(a)
#define DNX(a) DUMPNX(a)
#define DNP(a) DUMPNP(a)
#define DNA(a) DUMPNA(a)
#define DNF(a) DUMPNF(a)
#define DPC(a) DUMPC(a)
#define DPS(a) DUMPS(a)
#define DPSPP(a) DUMPSPP(a)
#define DPD(a) DUMPD(a)
#define DPU(a) DUMPU(a)
#define DPX(a) DUMPX(a)
#define DPP(a) DUMPP(a)
#define DPA(a) DUMPA(a)
#define DPF(a) DUMPF(a)

#define ARRAYDUMPC(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %c\n", xi, (a)[xi]);}
#define ARRAYDUMPS(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %s\n", xi, (a)[xi]);}
#define ARRAYDUMPD(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %2d\n", xi, (a)[xi]);}
#define ARRAYDUMPU(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %2u\n", xi, (a)[xi]);}
#define ARRAYDUMPX(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = 0x%X\n", xi, (u32)(a)[xi]);}
#define ARRAYDUMPX2(a,length) printf(#a" = ");for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf("%02X ", (u32)((a)[xi]));}printf("\n");
#define ARRAYDUMPX3(a,length) printf(#a" =\n");for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf("%02X ", (u32)((a)[xi]));if(xi%16==15){printf("\n");}}printf("\n");
#define ARRAYDUMPP(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %p\n", xi, (void *)(a)[xi]);}
#define ARRAYDUMPA(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = 0x%08X\n", xi, POINTER_TO_U32((a)[xi]));}

#define ARRAYDUMPX2_VERBOS(a,length) printf(#a" = ");for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf("0x%02X, ", (u32)((a)[xi]));}printf("\n");

#define PRINT_NEXT_LINE printf("\n");
#define NEWLINE         PRINT_NEXT_LINE

#define PRINT_FUNC      printf("%s\n",__func__);
#define PRINT_LINE      printf("%d\n",__LINE__);
#define PRINT_FILE      printf("%s\n",__FILE__);
#define PRINT_FUNC_LINE printf("%s() line:%d\n",__func__,__LINE__);
//#define PRINT_FUNC(1) PRINT_FUNC PRINT_NEXT_LINE
//#define PRINT_LINE(1) PRINT_LINE PRINT_NEXT_LINE
//#define PRINT_FILEn PRINT_FILE PRINT_NEXT_LINE
//#define PRLOC PRINT_FUNC_LINE

//#define pr(fmt, ...) printf(fmt, ##__VA_ARGS__)
//#define pn(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

//verbose version
//#define pv(fmt, ...) printf("[%4d]%s() : " fmt, __LINE__, __func__, ##__VA_ARGS__)

#define VARCLR(var) memset(&var,0,sizeof(var))
#define BLOCKCLR(p) memset(p,0,sizeof(*(p)))
#define RANGECLR(p,len) memset(p,0,len)

#ifndef ENABLE_DBG_MSG
#define ENABLE_DBG_MSG (1)
#endif

#ifndef ENABLE_ERR_MSG
#define ENABLE_ERR_MSG (1)
#endif

#if (ENABLE_DBG_MSG)
#define DBG_MSG(...) printf(##__VA_ARGS__)
#else
#define DBG_MSG(...)
#endif

#if (ENABLE_ERR_MSG)
#define DBG_ERR(...) printf(##__VA_ARGS__)
#else
#define DBG_ERR(...)
#endif

/* To close debug message (RE-DEFINE) */
/*
#ifdef ENABLE_DBG_MSG
#undef ENABLE_DBG_MSG
#undef DBG_MSG
#define ENABLE_DBG_MSG (1)
	#if (ENABLE_DBG_MSG)
	#define DBG_MSG(...) printf(##__VA_ARGS__)
	#else
	#define DBG_MSG(...)
	#endif
#endif
*/

#define REMOVE_UNUSED_WARNING(a) (void)(a)

#define _MY_MACROFUNCS_H_INCLUDED_
#endif//_MY_MACROFUNCS_H_INCLUDED_

