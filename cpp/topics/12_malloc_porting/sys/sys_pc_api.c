
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

//#include <string>
#include <string.h>

#include "sys_pc_api.h"

int syspc_file_search(const char *file_with_path, char out_buf[], int out_buf_max_len)
{
    // std::string file_with_path_cpp;
    // file_with_path_cpp = file_with_path;

    // FILE * pFile;
    // pFile = fopen (file_with_path_cpp.c_str(), "r");
    // if (pFile != NULL)
    // {
    //     fclose(pFile);

    //     if ((int)file_with_path_cpp.size() >= out_buf_max_len) {
    //         BASIC_ASSERT(0);
    //         return 1;
    //     }
        
    //     strcpy(out_buf, file_with_path_cpp.c_str());
    //     return 0;
    // }

    // file_with_path_cpp = "../" + file_with_path_cpp;
    // pFile = fopen (file_with_path_cpp.c_str(), "r");
    // if (pFile != NULL)
    // {
    //     fclose(pFile);

    //     if ((int)file_with_path_cpp.size() >= out_buf_max_len) {
    //         BASIC_ASSERT(0);
    //         return 1;
    //     }
        
    //     strcpy(out_buf, file_with_path_cpp.c_str());
    //     return 0;
    // }

    // file_with_path_cpp = "../" + file_with_path_cpp;
    // pFile = fopen (file_with_path_cpp.c_str(), "r");
    // if (pFile != NULL)
    // {
    //     fclose(pFile);

    //     if ((int)file_with_path_cpp.size() >= out_buf_max_len) {
    //         BASIC_ASSERT(0);
    //         return 1;
    //     }
        
    //     strcpy(out_buf, file_with_path_cpp.c_str());
    //     return 0;
    // }

    return 1;
}

#include <time.h>
clock_t gClock = 0;
void LibTime_StartClock(void)
{
	gClock = clock();
}

void LibTime_StopClock(void)
{
	gClock = clock() - gClock;
	//return gClock;
}

double LibTime_CalculateClock(void)
{
	return ((double)gClock)/CLOCKS_PER_SEC;
}

void LibTime_StopClock_ShowResult(void)
{
	gClock = clock() - gClock;
	printf ("It took me %4d clicks (%f seconds).", (int)gClock, ((double)gClock)/CLOCKS_PER_SEC);
}

void LibTime_StopClock_ShowResultMs(void)
{
	gClock = clock() - gClock;
	printf ("It took me %4d clicks (%f mili-seconds).", (int)gClock, ((double)gClock)/CLOCKS_PER_SEC*1000);
}

//#include <unistd.h> // unsigned int sleep(unsigned int seconds);   http://man7.org/linux/man-pages/man3/usleep.3.html

void LibOs_SleepSeconds(unsigned int seconds)
{
	//sleep(seconds);
}

void LibOs_SleepMiliSeconds(unsigned int miliSeconds)
{
	// int usleep(useconds_t usec);  The type useconds_t is an unsigned integral type capable of storing values at least in the range [0, 1,000,000]
	//usleep((useconds_t)(1000 * miliSeconds));
}

void LibOs_SleepMicroSeconds(unsigned int microSeconds)
{
	//usleep((useconds_t)microSeconds);
}