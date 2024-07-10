
#include <iostream>
#include <cstdint>
#include <memory>

int libfunc1(int a, int b)
{
    a = a * 10;
    b = b * 10;
    return a + b;
}

__attribute__((visibility ("default"))) void libfunc2()
{
    printf("abxx\n");
}
