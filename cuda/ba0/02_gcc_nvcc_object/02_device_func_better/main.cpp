#include <stdio.h>
#include "data.h"

extern void gpu(Data_t *db, int db_num);
extern void gpu2(Data2 *db, int db_num);

int main() {
    Data_t db[10];
    db[0].a = 1;
    db[0].b = 2;
    db[0].c = 3;
    printf("%d, %d, %d\n", db[0].a, db[0].b, db[0].c);
    gpu(db, 1);
    printf("%d, %d, %d, d = %d\n", db[0].a, db[0].b, db[0].c, db[0].d);


    Data2 objAry[10];
    objAry[0].a = 2;
    objAry[0].b = 3;
    objAry[0].c = 4;
    printf("%d, %d, %d\n", objAry[0].a, objAry[0].b, objAry[0].c);
    gpu2(objAry, 1);
    printf("%d, %d, %d, d = %d\n", objAry[0].a, objAry[0].b, objAry[0].c, objAry[0].d);

    return 0;
}