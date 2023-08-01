//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

int main() 
{
    {
        typedef char * PCHAR;
        PCHAR p,q;              // p, q are both char *

        //test
        char c = 'c';
        p = &c;
        q = &c;
        printf("*p=%c, *q=%c\n", *p, *q);
    }

    {
        char * p,q;             // p is char *, q is char

        //test
        char c = 'c';
        p = &c;
        q = c;
        printf("*p=%c, q=%c\n", *p, q);
    }

    {
        typedef char * a;  // a is a pointer to a char

        typedef a b();     // b is a function that returns
                           // a pointer to a char

        typedef b *c;      // c is a pointer to a function
                           // that returns a pointer to a char

        typedef c d();     // d is a function returning
                           // a pointer to a function
                           // that returns a pointer to a char

        typedef d *e;      // e is a pointer to a function 
                           // returning  a pointer to a 
                           // function that returns a 
                           // pointer to a char

        e var[10];         // var is an array of 10 pointers to 
                           // functions returning pointers to 
                           // functions returning pointers to chars.
    }

    {
        typedef struct tagPOINT
        {
            int x;
            int y;
        }POINT;

        POINT p; /* Valid C code */ /* Omit the use of "struct" */
    }

    return 0;
}
