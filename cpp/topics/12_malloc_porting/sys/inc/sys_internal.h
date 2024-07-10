#ifndef _SYS_INTERNAL_H_INCLUDED_
#define _SYS_INTERNAL_H_INCLUDED_

#include <stdint.h>
#include <stdio.h>
#include "linked_list.h"
#include "macros.h"

#ifdef _NO_OS_
#define SYS_MAGIC_NUM (6010)
#else //#ifdef _NO_OS_
#define SYS_MAGIC_NUM (6000)
#endif //#ifdef _NO_OS_

#define OUT

typedef struct {
    LList_Entry_t en;
    int instance[];
} smsg_t;

typedef void * EVENT_HANDLE_t;
typedef void * MUTEX_HANDLE_t;

#define OUT
#define RETURN_IF(retVal) if(retVal){return (retVal);}

int LibIPC_Event_Create(OUT EVENT_HANDLE_t *eventHdlPtr); // AUTO RESET EVENT !!
int LibIPC_Event_Destroy(EVENT_HANDLE_t eventHdl);
int LibIPC_Event_Set(EVENT_HANDLE_t eventHdl);
int LibIPC_Event_Wait(EVENT_HANDLE_t eventHdl);
int LibIPC_Event_BatchCreate(EVENT_HANDLE_t *eventHdlAry, uint32_t len);
int LibIPC_Event_BatchDestroy(EVENT_HANDLE_t *eventHdlAry, uint32_t len);
int LibIPC_Event_GetCounter(void);

int LibIPC_Mutex_Create(OUT MUTEX_HANDLE_t *mutexHdlPtr);
int LibIPC_Mutex_Destroy(MUTEX_HANDLE_t mutexHdl);
int LibIPC_Mutex_Lock(MUTEX_HANDLE_t mutexHdl);
int LibIPC_Mutex_Unlock(MUTEX_HANDLE_t mutexHdl);
int LibIPC_Mutex_GetCounter(void);

//typedef void *(*ThreadEntryFunc)(void *);
typedef void * THREAD_HANDLE_t;
int LibThread_NewHandle(OUT THREAD_HANDLE_t *threadHdlPtr, uint32_t priority /* = TPRI_DEFAULT */);
int LibThread_Create(THREAD_HANDLE_t threadHdl, ThreadEntryFunc entry, void *arg /* = NULL */);
int LibThread_WaitThread(THREAD_HANDLE_t threadHdl);
int LibThread_DestroyHandle(THREAD_HANDLE_t threadHdl);
int LibThread_GetCounter(void);

#endif//_SYS_INTERNAL_H_INCLUDED_
