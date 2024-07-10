#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "sys.h"
#include "libmt.h"
#include "libmem.h"

int Lib_IsMT(void)
{
    return 1;
}

int LibMT_UtilMutex_Init(LibMT_UtilMutex_t *mutex)
{
    if (Lib_IsMT()) {
        int retVal;
        BASIC_ASSERT(mutex->handle == NULL);
        ASSERT_CHK( retVal, LibIPC_Mutex_Create(&(mutex->handle)) );
    }
    return 0;
}

int LibMT_UtilMutex_Uninit(LibMT_UtilMutex_t *mutex)
{
    if (Lib_IsMT()) {
        int retVal;
        BASIC_ASSERT(mutex->handle != NULL);
        ASSERT_CHK( retVal, LibIPC_Mutex_Destroy(mutex->handle) );
        mutex->handle = NULL;
    }
    return 0;
}

int LibMT_UtilMutex_Lock(LibMT_UtilMutex_t *mutex)
{
    if (Lib_IsMT()) {
        int retVal;
        BASIC_ASSERT(mutex->handle != NULL);
        ASSERT_CHK( retVal, LibIPC_Mutex_Lock(mutex->handle) );
    }
    return 0;
}

int LibMT_UtilMutex_Unlock(LibMT_UtilMutex_t *mutex)
{
    if (Lib_IsMT()) {
        int retVal;
        BASIC_ASSERT(mutex->handle != NULL);
        ASSERT_CHK( retVal, LibIPC_Mutex_Unlock(mutex->handle) );
    }
    return 0;
}

#define SHOW_RACE_CONDITION ( 0 )

volatile int race_condition = 100;
LibMT_UtilMutex_t gTestUtilMutex1;

LibMT_UtilMutex_t gLibMT_MsgLock;
DLList_Head_t gLibMT_MsgHead;
#define LIB_MT_PREALLOCATE_MSG_NUM ( 16 )
#define LIB_MT_MSG_LOCK   LibMT_UtilMutex_Lock(&gLibMT_MsgLock);
#define LIB_MT_MSG_UNLOCK LibMT_UtilMutex_Unlock(&gLibMT_MsgLock);

LibMT_UtilMutex_t gLibMT_ThreadLock;
DLList_Head_t gLibMT_ThreadHead;
#define LIB_MT_THREAD_LOCK   LibMT_UtilMutex_Lock(&gLibMT_ThreadLock);
#define LIB_MT_THREAD_UNLOCK LibMT_UtilMutex_Unlock(&gLibMT_ThreadLock);

LibMT_UtilMutex_t gLibMT_Mutex_Array[MUTEX_TOTAL_NUM];
int LibMT_Mutex_Lock(LIBMT_MUTEX_t index)
{
    return LibMT_UtilMutex_Lock(&(gLibMT_Mutex_Array[index]));
}

int LibMT_Mutex_Unlock(LIBMT_MUTEX_t index)
{
    return LibMT_UtilMutex_Unlock(&(gLibMT_Mutex_Array[index]));
}

int LibMT_Init(void)
{
    LibMT_UtilMutex_Init(&gLibMT_MsgLock);
    LibMT_UtilMutex_Init(&gLibMT_ThreadLock);

    FOREACH_I(MUTEX_TOTAL_NUM) {
        LibMT_UtilMutex_Init(&(gLibMT_Mutex_Array[i]));
    }


    DLLIST_HEAD_RESET(&gLibMT_ThreadHead);

    return 0;
}

int LibMT_Uninit(void)
{
    LibMT_UtilMutex_Uninit(&gLibMT_MsgLock);
    LibMT_UtilMutex_Uninit(&gLibMT_ThreadLock);

    FOREACH_I(MUTEX_TOTAL_NUM) {
        LibMT_UtilMutex_Uninit(&(gLibMT_Mutex_Array[i]));
    }

    return 0;
}