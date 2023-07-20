//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

int main() 
{
    int x = ALL_VER;
    int *px = &x;

    printf("%d, %d, %p\n", x, *px, px);
    return 0;
}