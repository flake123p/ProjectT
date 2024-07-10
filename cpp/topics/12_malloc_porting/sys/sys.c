
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "sys.h"
#include "sys_internal.h"
//#include "pupil.h"
//#include "linked_list.h"

#include <assert.h>

//TODO: Using struct/array to manager task info.

/*
	sys util
*/
static int g_sys_print_enable = 1;
void sys_print_control(int en)
{
    g_sys_print_enable = en;
}

void sys_print(const char *format, ...)
{
    if (g_sys_print_enable == 0)
        return;

    //UART??
	va_list ap;

	va_start(ap, format);
	vprintf(format, ap);
	va_end(ap);
}

void sys_assert(int expression, const char *file, int line)
{
    if (!expression)
        printf("[%d]Assert in file: %s, line: %d\n", expression, file, line);
    assert(expression);
}

// This version of cmd api is pthread + task msg.
extern msg_task_handle_t g_task_hi_hdl;
extern msg_task_handle_t g_task_lo_hdl;
extern msg_task_handle_t g_tool_cmd_hdl;
extern msg_task_handle_t g_tool_evt_hdl;

int cmd_req_send(void *msg)
{
    BASIC_ASSERT(g_tool_cmd_hdl != NULL);
    MSGTASK_MSG_SEND(g_task_lo_hdl, msg);
    return 0;
}

int cmd_res_send(void *msg)
{
    BASIC_ASSERT(g_tool_cmd_hdl != NULL);
    MSGTASK_MSG_SEND(g_tool_cmd_hdl, msg);
    return 0;
}

int event_send(void *msg)
{
    BASIC_ASSERT(g_tool_evt_hdl != NULL);
    MSGTASK_MSG_SEND(g_tool_evt_hdl, msg);
    return 0;
}

void prloc(const char *file, int line)
{
    printf("file: %s, line %d\n", file, line);
}

// MUTEX_HANDLE_t g_sysmemlock = NULL;
// int sys_memlock_init()
// {
// 	int ret;
// 	ret = LibIPC_Mutex_Create(&g_sysmemlock);
//     BASIC_ASSERT(ret == 0);
// 	return ret;
// }

// int sys_memlock_uninit()
// {
// 	int ret = 0;
// 	if (g_sysmemlock != NULL) {
// 		ret = LibIPC_Mutex_Destroy(g_sysmemlock);
// 		BASIC_ASSERT(ret == 0);
// 	}
// 	g_sysmemlock = NULL;
// 	return ret;
// }

// int sys_memlock_get()
// {
// 	BASIC_ASSERT(g_sysmemlock != NULL);
// 	return LibIPC_Mutex_Lock(g_sysmemlock);
// }

// int sys_memlock_release()
// {
// 	BASIC_ASSERT(g_sysmemlock != NULL);
// 	return LibIPC_Mutex_Unlock(g_sysmemlock);
// }

int sys_magic_number(void)
{
	return SYS_MAGIC_NUM;
}

void sys_internal_counter_clear_check(void)
{
    BASIC_ASSERT(LibThread_GetCounter() == 0);
    BASIC_ASSERT(LibIPC_Event_GetCounter() == 0);
    BASIC_ASSERT(LibIPC_Mutex_GetCounter() == 0);
}

#define SYS_CYCLE_TOTAL (60)
int sys_cycle_def;
int sys_abs_cycle[SYS_CYCLE_TOTAL] = {0};
int sys_rltv_cycle[SYS_CYCLE_TOTAL] = {0};
int sys_max_cycle_idx = 0;

int sys_prof_get_cycle()
{
#define rd_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })
    return rd_csr(0xB00);
}

void sys_prof_reset_cycle()
{
    sys_cycle_def = sys_prof_get_cycle();
}

void sys_prof_probe_cycle(int idx)
{
    BASIC_ASSERT(idx < SYS_CYCLE_TOTAL);

    int curr = sys_prof_get_cycle();
    // sys_cycle[idx] = curr;
    if (idx == 0) {
        sys_abs_cycle[idx] = curr;
        sys_rltv_cycle[idx] = 0;
    } else {
        sys_abs_cycle[idx] = curr;
        sys_rltv_cycle[idx] = curr - sys_abs_cycle[idx-1];
        sys_max_cycle_idx = idx;
    }
}
