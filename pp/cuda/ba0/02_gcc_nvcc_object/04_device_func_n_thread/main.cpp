#include <stdio.h>
#include "hpc.h"

// extern void gpu(Data_t *db, int db_num);
// extern void gpu2(Data2 *db, int db_num);

int main() {
    EUnit objAry[10];
    for (int i = 0; i < 10; i++) {
        objAry[i].a[0] = 1 + i;
        objAry[i].a[1] = 2 + i;
        objAry[i].a[2] = 1000;
        objAry[i].a[3] = 0;
    }

    runThreads(objAry, 10);
    printf("runThreads:\n");

    for (int i = 0; i < 10; i++) {
        printf("result %d = %d\n", i, objAry[i].a[3]);
    }

    for (int i = 0; i < 10; i++) {
        objAry[i].a[0] = 1 + i;
        objAry[i].a[1] = 2 + i;
        objAry[i].a[2] = 1000;
        objAry[i].a[3] = 0;
    }

    runGpu(objAry, 10);
    printf("runGpu:\n");

    for (int i = 0; i < 10; i++) {
        printf("result %d = %d\n", i, objAry[i].a[3]);
    }

    return 0;
}