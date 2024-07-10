#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int foo(uint32_t *ary)
{
    ary[0] = ary[2] + ary[3];
    ary[1] = ary[2] * ary[3];
    return 101;
}
/*
    Dump & Var/Mem read write

    -exec info registers
        : dump registers via VSCode

    -exec dump ihex memory mdump.txt 0x55555556aeb0 0x55555556aed0
        : failed on VSCode (just browse by the hitting the memory view icon aside)

    -exec p abxx

    -exec set variable abxx = 22
    
    -exec set *((unsigned long)0x55555556aeb0) = 11
*/


int main()
{
    uint32_t abxx = 2;
    uint32_t *ary = (uint32_t *)malloc(sizeof(uint32_t)*4);

    printf("Hello a01\n");
    ary[2] = abxx;
    ary[3] = 7;
    foo(ary);

    printf("%u, %u\n", ary[0], ary[1]);

    free(ary);
    return 0;
}

