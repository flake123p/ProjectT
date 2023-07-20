
#include <iostream>
#include <cstdint>
#include <memory>
#include "all.hpp"

int main() 
{
    int x = ALL_VER;
    int *px = &x;

    printf("%d, %d, %p\n", x, *px, px);
    return 0;
}