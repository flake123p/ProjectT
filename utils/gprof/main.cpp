#include <stdio.h>

volatile int ary[1024];

void testImpl() {
    for (int i = 0; i < 1024; i++) {
        ary[i] = i;
    }
}

void test() {
    for (int i = 0; i < 1000000; i++) {
        testImpl();
    }
}

int main() {
    test();
    return 0;
}
