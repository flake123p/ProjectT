#ifndef _SYS_PC_API_H_INCLUDED_
#define _SYS_PC_API_H_INCLUDED_

/*
    PC API, not for embedded or IC environment.
*/
#include "sys.h"
int syspc_file_search(const char *file_with_path, char out_buf[], int out_buf_max_len);
#define OLD_SYS_FILE_SEARCH(out,in) syspc_file_search(in,out,ARRAY_LEN(out))

void LibTime_StartClock(void);
void LibTime_StopClock(void);
double LibTime_CalculateClock(void);
void LibTime_StopClock_ShowResult(void);
void LibTime_StopClock_ShowResultMs(void);

#endif//_SYS_PC_API_H_INCLUDED_
