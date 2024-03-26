#include <stdio.h>
#include "data.h"

extern void gpu(Data_t *db, int db_num);

int main() {
    Data_t db[10];
    db[0].a = 1;
    db[0].b = 2;
    db[0].c = 3;
    printf("%d, %d, %d\n", db[0].a, db[0].b, db[0].c);
    gpu(db, 1);
    printf("%d, %d, %d, d = %d\n", db[0].a, db[0].b, db[0].c, db[0].d);

    return 0;
}