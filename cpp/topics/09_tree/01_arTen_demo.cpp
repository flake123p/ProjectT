#include "_list.h"
#include "_tree.h"
#include "_arTen.h"

int main()
{
    {
        printf("\nDemo1:\n");
        class ArTen<void *> at({2,3,4});

        at.dump();
    }
    {
        printf("\nDemo2:\n");
        class ArTen<int> at({2,3});
        //at.dump();
        int val = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                at.set({i,j}, val);
                val++;
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                printf("i:%d, j:%d, val:%d\n", i, j, at.get({i, j}));
            }
        }
        at.dump_array([](int idx, int cur_val) {
            printf("idx = %d, val = %d", idx, cur_val);
        });
        at.travers_array([](int idx, int *inst) {
            idx = idx;
            *inst = 2266;
        });
        at.dump_array([](int idx, int cur_val) {
            printf("idx = %d, val = %d", idx, cur_val);
        });
    }
}