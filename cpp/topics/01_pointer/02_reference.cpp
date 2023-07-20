//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

uint8_t gMem[] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1f,
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,  0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3f,
};

int main() 
{
    int x = 123;
    int y = 456;

    int &ref = x;

    printf("[1]Before:\n");
    printf("*x  = %d\n", x);
    printf("*ref= %d\n", ref);

    ref = y;
    printf("[1]After:\n");
    printf("*x  = %d\n", x);
    printf("*ref= %d\n", ref);
    printf("\n");

    x = 123;
    y = 456;
    int *p0 = &x;
    int **p1;       //  p1 is a pointer   to a pointer   to an int.
    int *&p2 = p0;  //  p2 is a reference to a pointer   to an int.
    //int &*p3;  //  ERROR: Pointer    to a reference is illegal.
    //int &&p4;  //  ERROR: Reference  to a reference is illegal.


    printf("[2]Before:\n");
    printf("*p0 = %d\n", *p0);
    printf("*p2 = %d\n", *p2);

    ref = y;
    p0 = &y;
    printf("[2]After:\n");
    printf("*p0 = %d\n", *p0);
    printf("*p2 = %d\n", *p2);

    return 0;
}
