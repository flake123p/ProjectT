#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "sys.h"
#include "sys_internal.h"
#include "libmem.h"
#include "libmt.h"

void mem_init()
{
    MM_INIT();
    //LibMT_Init();
}

void mem_uninit()
{
    //LibMT_Uninit();
    MM_UNINIT();
}

void *mem_alloc(size_t size)
{
    void *ret;
    ret = MM_ALLOC(size);
    return ret;
}

int mem_free(void *ptr)
{
    MM_FREE(ptr);
    return 0;
}

void *dummy_msg_alloc(uint32_t size)
{
    {
        LOCK_MEM;
    }
    void *ret = NULL;
    smsg_t *ptr;
    uint32_t size_with_padding = size + sizeof(smsg_t);
    ptr = (smsg_t *)mem_alloc(size_with_padding);
    ret = ptr->instance;
    //printf("1 %p\n", ptr);
    //printf("2 %p\n", ptr->instance);
    {
        UNLOCK_MEM;
    }
    return ret;
}

int dummy_msg_free(void *ptr)
{
    {
        LOCK_MEM;
    }
    void *entry;
    entry = (void *)STRUCT_ENTRY(ptr, smsg_t, instance);
    //printf("3 %p\n", ptr);
    //printf("4 %p\n", entry);
    mem_free(entry);
    {
        UNLOCK_MEM;
    }
    return 0;
}

void *dummy_alloc_tracer(uint32_t size, const char *file_str, int line)
{
    {
        LOCK_MEM;
    }
    void *ret = NULL;
    smsg_t *ptr;
    uint32_t size_with_padding = size + sizeof(smsg_t);
    //printf("11 %d\n", size);
    //printf("22 %d (%d)\n", size, sizeof(smsg_t));
    REMOVE_UNUSED_WARNING(file_str);
    REMOVE_UNUSED_WARNING(line);
    ptr = (smsg_t *)MM_ALLOC3(size_with_padding, file_str, line);
    ret = ptr->instance;
    {
        UNLOCK_MEM;
    }
    return ret;
}

void *dummy_msg_alloc_tracer(uint32_t size, const char *file_str, int line)
{
    {
        LOCK_MEM;
    }
    void *ret = NULL;
    smsg_t *ptr;
    uint32_t size_with_padding = size + sizeof(smsg_t);
    //printf("11 %d\n", size);
    //printf("22 %d (%d)\n", size, sizeof(smsg_t));
    REMOVE_UNUSED_WARNING(file_str);
    REMOVE_UNUSED_WARNING(line);
    ptr = (smsg_t *)MM_ALLOC3(size_with_padding, file_str, line);
    ret = ptr->instance;

    {
        UNLOCK_MEM;
    }
    return ret;
}
