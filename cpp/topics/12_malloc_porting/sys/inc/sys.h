#ifndef _SYS_H_INCLUDED_
#define _SYS_H_INCLUDED_

#include "linked_list.h"
#include "macros.h"
#include "type.h"

#include <stdio.h>  // flake: remove this

#include "libmem.h"

void *dummy_msg_alloc(uint32_t size);
int dummy_msg_free(void *ptr);
void *dummy_alloc_tracer(uint32_t size, const char *file_str, int line);
void *dummy_msg_alloc_tracer(uint32_t size, const char *file_str, int line);
#define DUMMY_ALLOC(size) dummy_alloc_tracer(size,__FILE__,__LINE__)
#define DUMMY_ALLOC_MSG(size) dummy_msg_alloc_tracer(size,__FILE__,__LINE__)
#define DUMMY_FREE dummy_msg_free


#ifdef  _NO_MEM_TRACER_ 
#define MEM_ALLOC mem_alloc
#define MEM_FREE mem_free
#define SBOX_MEM_ALLOC malloc
#define SBOX_MEM_FREE free
#else
#include "libmem.h"
#define MEM_ALLOC MM_ALLOC
#define MEM_FREE MM_FREE
#define SBOX_MEM_ALLOC MM_ALLOC
#define SBOX_MEM_FREE MM_FREE
#endif

void mem_init();
void mem_uninit();
void *mem_alloc(size_t size);
int mem_free(void *ptr);

void sys_print_control(int en);
void sys_print(const char *format, ...);

void sys_assert(int expression, const char *file, int line);
//TODO: disable compiling switch
#ifdef _NO_ASSERT_
    #define BASIC_ASSERT(...)
    #define CALLER_ASSERT3(...)
    #define CALLER_PARA3
    #define CALLER_ARG3
#else
    #define BASIC_ASSERT(a) sys_assert(a, __FILE__, __LINE__)
    //NOTE: Prepare caller_file & caller_line before use this
    #define CALLER_ASSERT3(a) if(a){;}else{sys_print(" CALLER asserted: in %s(), line %ld\n",caller_file,caller_line);BASIC_ASSERT(0);}
    #define CALLER_PARA3 , const char * caller_file, unsigned long caller_line
    #define CALLER_ARG3 ,__FUNCTION__,__LINE__
#endif

#define ASSERT_IF(retVal) if(retVal){BASIC_ASSERT(0);};
#define ASSERT_CHK(a, b)  a=b;ASSERT_IF(a)
#define RETURN_IF(retVal) if(retVal){return (retVal);}
#define RETURN_CHK(a, b)  a=b;RETURN_IF(a)
#define RETURN_WHEN(condition, retVal) if(condition){return (retVal);}

#ifdef _NO_PRINT_
    #define BASIC_PRINT(...)
#else
    #define BASIC_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#define BASIC_TEST_CASE(test_func) \
    do { \
        BASIC_PRINT("[%s] START\n", #test_func); \
        mem_init(); \
        test_func(); \
        mem_uninit(); \
        BASIC_PRINT("[%s] PASS\n", #test_func); \
    } while(0);


#define BASIC_TEST_CASE2(test_func) \
    do { \
        int ret; \
        BASIC_PRINT("[%s] START\n", #test_func); \
        mem_init(); \
        ret = test_func(); \
        mem_uninit(); \
        if (ret) { \
            BASIC_PRINT("[%s] FAILED (%d)\n", #test_func, ret); \
        } else { \
            BASIC_PRINT("[%s] PASS\n", #test_func); \
        } \
    } while(0);


// #define DUMPNS(a) printf(#a " = %s\n", a)
// #define DUMPND(a) printf(#a " = %2d\n", (int)(a))
// #define DUMPNU(a) printf(#a " = %2u\n", (unsigned int)a)
// #define DUMPNX(a) printf(#a " = 0x%X\n", (unsigned int)(a))
// #define DUMPNP(a) printf(#a " = %p\n", (void *)(a))
// #define DNS(a) DUMPNS(a)
// #define DND(a) DUMPND(a)
// #define DNU(a) DUMPNU(a)
// #define DNX(a) DUMPNX(a)
// #define DNP(a) DUMPNP(a)

typedef void *(*ThreadEntryFunc)(void *);

typedef void * msg_task_handle_t; 

// USE MACRO PLEASE
int msg_task_new_handle(OUT msg_task_handle_t *hdl_ptr, uint32_t priority);
int msg_task_destory_handle(msg_task_handle_t hdl);
int msg_task_create(msg_task_handle_t hdl, ThreadEntryFunc cb, void *arg);
int msg_task_wait(msg_task_handle_t hdl);

int msg_task_new_handle_no_os(OUT msg_task_handle_t *hdl_ptr, uint32_t priority);
int msg_task_destory_handle_no_os(msg_task_handle_t hdl);

#ifdef _NO_OS_
#define MSGTASK_NEW_HANDLE(hdl_ptr,priority) msg_task_new_handle_no_os(hdl_ptr,priority)
#define MSGTASK_DESTROY_HANDLE(hdl) msg_task_destory_handle_no_os(hdl)
#define MSGTASK_CRREATE(hdl,cb,arg)
#define MSGTASK_WAIT(hdl)

#else // #ifdef _NO_OS_
#define MSGTASK_NEW_HANDLE(hdl_ptr,priority) msg_task_new_handle(hdl_ptr,priority)
#define MSGTASK_DESTROY_HANDLE(hdl) msg_task_destory_handle(hdl)
#define MSGTASK_CRREATE(hdl,cb,arg) msg_task_create(hdl,cb,arg)
#define MSGTASK_WAIT(hdl) msg_task_wait(hdl)

#endif // #ifdef _NO_OS_



// USE MACRO PLEASE
void msg_send(msg_task_handle_t hdl, void *msg);
void *msg_receive(msg_task_handle_t hdl);
void msg_wait(msg_task_handle_t hdl);
void *msg_wait_and_receive(msg_task_handle_t hdl);

void msg_send_no_os(msg_task_handle_t hdl, void *msg);
void *msg_receive_no_os(msg_task_handle_t hdl);

#ifdef _NO_OS_
#define MSGTASK_MSG_SEND(hdl,msg) msg_send_no_os(hdl,msg) // Please implement MSGTASK_MSG_SEND on upper layer
#define MSGTASK_MSG_RECV(hdl) msg_receive_no_os(hdl)
#define MSGTASK_MSG_WAIT(hdl)
#define MSGTASK_MSG_WAIT_n_RECV(hdl) msg_receive_no_os(hdl)

#else // #ifdef _NO_OS_
#define MSGTASK_MSG_SEND(hdl,msg) msg_send(hdl,msg)
#define MSGTASK_MSG_RECV(hdl) msg_receive(hdl)
#define MSGTASK_MSG_WAIT(hdl) msg_wait(hdl)
#define MSGTASK_MSG_WAIT_n_RECV(hdl) msg_wait_and_receive(hdl)

#endif // #ifdef _NO_OS_

typedef void (*state_machine_t)(void);
void register_no_os_sm(state_machine_t sm);

// cmd req src = tool
// cmd req dst = task_lo
// cmd res src = task_lo
// cmd res dst = tool
// evt src     = task_hi
// evt dst     = tool
int cmd_internal_interface_config(msg_task_handle_t task_hi_hdl, msg_task_handle_t task_lo_hdl);
int cmd_external_interface_config(msg_task_handle_t tool_cmd_hdl, msg_task_handle_t tool_evt_hdl);
int cmd_req_send(void *msg);
int cmd_res_send(void *msg);
int event_send(void *msg);

#define PRLOC prloc(__FILE__, __LINE__);
void prloc(const char *file, int line);

// int sys_memlock_init();
// int sys_memlock_uninit();
// int sys_memlock_get();
// int sys_memlock_release();
//#define LOCK_MEM sys_memlock_get()
//#define UNLOCK_MEM sys_memlock_release()
#define LOCK_MEM
#define UNLOCK_MEM

int sys_magic_number(void);

void sys_internal_counter_clear_check(void);

//#define REMOVE_UNUSED_WARNING(a) (a=a)
//#define REMOVE_UNUSED_WARNING(a) (void)(a)
#include "My_MacroFuncs.h"

void LibOs_SleepSeconds(unsigned int seconds);
void LibOs_SleepMiliSeconds(unsigned int miliSeconds);
void LibOs_SleepMicroSeconds(unsigned int microSeconds);

#define USING_FREERTOS_HEAP_4 (1)
#define USING_FREERTOS_HEAP_5 (0)
typedef long BaseType_t;
void * _pvPortMalloc( size_t xWantedSize );
void _vPortFree( void * pv );

#ifdef _USING_MY_MALLOC_
extern void *malloc (size_t);
extern void free (void *);
extern void *realloc (void *, size_t);
extern void *calloc (size_t, size_t);
#endif

#ifdef _USING_FLOAT_32_
#define _SIN  sinf
#define _COS  cosf
#define _TAN  tanf
#define _ASIN asinf
#define _ACOS acosf
#define _ATAN atanf
#define _ATAN2 atan2f
#define _EXP  expf
#define _SQRT sqrtf
#define _POW  powf
#else
#define _SIN  sin
#define _COS  cos
#define _TAN  tan
#define _ASIN asin
#define _ACOS acos
#define _ATAN atan
#define _ATAN2 atan2
#define _EXP  exp
#define _SQRT sqrt
#define _POW  pow
#endif

#ifndef null
#define null NULL
#endif

//For argument
#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

#ifndef IO
#define IO
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif
#ifndef ushort
typedef unsigned short ushort;
#endif
#ifndef uint
typedef unsigned int uint;
#endif
#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef u8
typedef uint8_t u8;
#endif
#ifndef u16
typedef uint16_t u16;
#endif
#ifndef u32
typedef uint32_t u32;
#endif
#ifndef u64
typedef uint64_t u64;
#endif

#ifndef s8
typedef int8_t s8;
#endif
#ifndef s16
typedef int16_t s16;
#endif
#ifndef s32
typedef int32_t s32;
#endif
#ifndef s64
typedef int64_t s64;
#endif

int sys_prof_get_cycle();
void sys_prof_reset_cycle();
void sys_prof_probe_cycle(int idx);

#define _likely(x)       __builtin_expect(!!(x), 1)
#define _unlikely(x)     __builtin_expect(!!(x), 0)

#endif//_SYS_H_INCLUDED_
