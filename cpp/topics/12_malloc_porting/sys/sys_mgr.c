
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "sys.h"
#include "sys_internal.h"
#include "libmem.h"
#include "libmt.h"

state_machine_t g_sm_no_os = NULL;
void register_no_os_sm(state_machine_t sm)
{
    g_sm_no_os = sm;
}

typedef struct {
    void *thre;
    EVENT_HANDLE_t q_evt;
    MUTEX_HANDLE_t q_mutex;
    LList_Head_t q_head;
} msg_task_t;

int msg_task_new_handle(msg_task_handle_t *hdl_ptr, uint32_t priority)
{
    int ret = 0;

    //TODO...
    msg_task_t *task = (msg_task_t *)mem_alloc(sizeof(msg_task_t));
    BASIC_ASSERT(task != NULL);

    ret = LibThread_NewHandle(&task->thre, priority);
    BASIC_ASSERT(ret == 0);
    BASIC_ASSERT(task->thre != NULL);
    if(ret) return ret;

    ret = LibIPC_Event_Create(&task->q_evt);
    BASIC_ASSERT(ret == 0);
    BASIC_ASSERT(task->q_evt != NULL);
    if(ret) return ret;

    ret = LibIPC_Mutex_Create(&task->q_mutex);
    BASIC_ASSERT(ret == 0);
    BASIC_ASSERT(task->q_mutex != NULL);
    if(ret) return ret;

    LLIST_HEAD_RESET(&task->q_head);

    *hdl_ptr = (msg_task_handle_t)task;

    return 0;
}

int msg_task_new_handle_no_os(msg_task_handle_t *hdl_ptr, uint32_t priority)
{
    msg_task_t *task = (msg_task_t *)mem_alloc(sizeof(msg_task_t));
    BASIC_ASSERT(task != NULL);

    REMOVE_UNUSED_WARNING(priority);

    LLIST_HEAD_RESET(&task->q_head);

    *hdl_ptr = (msg_task_handle_t)task;
    return 0;
}

int msg_task_destory_handle(msg_task_handle_t hdl)
{
    int ret = 0;
    msg_task_t *task = (msg_task_t *)hdl;

    LibThread_DestroyHandle(task->thre);

    LibIPC_Event_Destroy(task->q_evt);

    LibIPC_Mutex_Destroy(task->q_mutex);

    mem_free(hdl);

    return ret;
}

int msg_task_destory_handle_no_os(msg_task_handle_t hdl)
{
    int ret = 0;

    mem_free(hdl);

    return ret;
}

int msg_task_create(msg_task_handle_t hdl, ThreadEntryFunc cb, void *arg)
{
    msg_task_t *task = (msg_task_t *)hdl;
    return LibThread_Create(task->thre, cb, arg);
}

int msg_task_wait(msg_task_handle_t hdl)
{
    msg_task_t *task = (msg_task_t *)hdl;
    return LibThread_WaitThread(task->thre);
}

void msg_send(msg_task_handle_t hdl, void *msg)
{
    msg_task_t *task = (msg_task_t *)hdl;

    //printf("1 %p\n", msg);
    //printf("2 %p\n", STRUCT_ENTRY(msg, smsg_t, instance));

    // lock linked list
    LibIPC_Mutex_Lock(task->q_mutex);

    //printf("1 %d\n", msg->id);
    _WT(STRUCT_ENTRY(msg, smsg_t, instance)->en.prev, NULL);
    _WT(STRUCT_ENTRY(msg, smsg_t, instance)->en.next, NULL);

    //LLIST_INSERT_LAST(&task->q_head, msg);
    LLIST_INSERT_LAST(&task->q_head, STRUCT_ENTRY(msg, smsg_t, instance));

    LibIPC_Mutex_Unlock(task->q_mutex);

    // wake thread
    LibIPC_Event_Set(task->q_evt);
}

void msg_send_no_os(msg_task_handle_t hdl, void *msg)
{
    msg_task_t *task = (msg_task_t *)hdl;
    LLIST_INSERT_LAST(&task->q_head, STRUCT_ENTRY(msg, smsg_t, instance));

    if (g_sm_no_os != NULL) {
        (*g_sm_no_os)();
    }
}

void *msg_receive(msg_task_handle_t hdl)
{
    msg_task_t *task = (msg_task_t *)hdl;
    smsg_t *smsg;

    // lock linked list
    LibIPC_Mutex_Lock(task->q_mutex);

    if (LLIST_IS_EMPTY(&task->q_head)) {
        LibIPC_Mutex_Unlock(task->q_mutex);
        return NULL;
    } else {
        smsg = (smsg_t *)LLIST_FIRST(&task->q_head);
        LLIST_REMOVE_FIRST_SAFELY(&task->q_head);
    }

    LibIPC_Mutex_Unlock(task->q_mutex);

    return smsg->instance;
}

void *msg_receive_no_os(msg_task_handle_t hdl)
{
    msg_task_t *task = (msg_task_t *)hdl;
    smsg_t *smsg;

    if (LLIST_IS_EMPTY(&task->q_head)) {
        return NULL;
    } else {
        smsg = (smsg_t *)LLIST_FIRST(&task->q_head);
        LLIST_REMOVE_FIRST_SAFELY(&task->q_head);
    }

    return smsg->instance;
}

void msg_wait(msg_task_handle_t hdl)
{
    msg_task_t *task = (msg_task_t *)hdl;
    LibIPC_Event_Wait(task->q_evt);
}

void *msg_wait_and_receive(msg_task_handle_t hdl)
{
    void *msg;

    while (1) {
        msg = MSGTASK_MSG_RECV(hdl);
        if (msg) {
            return msg;
        } else {
            MSGTASK_MSG_WAIT(hdl);
        }
    }
}

// for sys.c
msg_task_handle_t g_task_hi_hdl = NULL;
msg_task_handle_t g_task_lo_hdl = NULL;
msg_task_handle_t g_tool_cmd_hdl = NULL;
msg_task_handle_t g_tool_evt_hdl = NULL;

int cmd_internal_interface_config(msg_task_handle_t task_hi_hdl, msg_task_handle_t task_lo_hdl)
{
    g_task_hi_hdl = task_hi_hdl;
    g_task_lo_hdl = task_lo_hdl;
    return 0;
}

int cmd_external_interface_config(msg_task_handle_t tool_cmd_hdl, msg_task_handle_t tool_evt_hdl)
{
    g_tool_cmd_hdl = tool_cmd_hdl;
    g_tool_evt_hdl = tool_evt_hdl;
    return 0;
}
