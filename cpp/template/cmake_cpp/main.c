// A simple program that computes the square root of a number
#include <stdio.h>
#include <stdint.h>

volatile int x = 10;

int main(int argc, char* argv[])
{
    x = 8;
    printf("Hello World! %d\n", x);
    x = 10;
    printf("Hello World! %d\n", x);

    return 0;
}
