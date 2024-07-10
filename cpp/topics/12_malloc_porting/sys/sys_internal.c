#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "sys.h"
#include "sys_internal.h"
//#include "pupil.h"
//#include "linked_list.h"

//Linux platform
// #include <fcntl.h>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <pthread.h>
#include <assert.h>

int LibIPC_Event_Create(OUT EVENT_HANDLE_t *eventHdlPtr)
{
	return 0;
}

int LibIPC_Event_Destroy(EVENT_HANDLE_t eventHdl)
{
	return 0;
}

int LibIPC_Event_Set(EVENT_HANDLE_t eventHdl)
{
	return 0;
}

int LibIPC_Event_Wait(EVENT_HANDLE_t eventHdl)
{
	return 0;
}

int LibIPC_Event_GetCounter(void)
{
	return 0;
}

int LibIPC_Mutex_Create(OUT MUTEX_HANDLE_t *mutexHdlPtr)
{
	return 0;
}

int LibIPC_Mutex_Destroy(MUTEX_HANDLE_t mutexHdl)
{
	return 0;
}

int LibIPC_Mutex_Lock(MUTEX_HANDLE_t mutexHdl)
{
	return 0;
}

int LibIPC_Mutex_Unlock(MUTEX_HANDLE_t mutexHdl)
{
	return 0;
}

int LibIPC_Mutex_GetCounter(void)
{
	return 0;
}

int LibThread_NewHandle(OUT THREAD_HANDLE_t *threadHdlPtr, uint32_t priority /* = TPRI_DEFAULT */)
{
	return 0;
}

int LibThread_Create(THREAD_HANDLE_t threadHdl, ThreadEntryFunc entry, void *arg /* = NULL */)
{
	return 0;
}

int LibThread_WaitThread(THREAD_HANDLE_t threadHdl)
{
	return 0;
}

int LibThread_DestroyHandle(THREAD_HANDLE_t threadHdl)
{
	return 0;
}

int LibThread_GetCounter(void)
{
	return 0;
}