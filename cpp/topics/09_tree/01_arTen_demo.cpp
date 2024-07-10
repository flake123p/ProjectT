#include "_list.h"
//#include "_tree.h"
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
        printf("Elem 0: %d\n", at(0));
        printf("Elem 1: %d\n", at(1));

        at.travers_array([](int idx, int *inst) {
            printf("idx = %d, val = %d\n", idx, *inst);
        });
        at.travers_array([](int idx, int *inst) {
            idx = idx;
            *inst = 2266;
        });
        at.travers_array([](int idx, int *inst) {
            printf("idx = %d, val = %d\n", idx, *inst);
        });
    }
}