//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

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
