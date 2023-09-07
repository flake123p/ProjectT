//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

int main() 
{
    {
        const int n=5;   // const int, preferred personally
        int const m=10;  // const int

        const int *p; // pointer to const int
        int const *q; // pointer to const int
        q = &n;
    }

    {
        int n = 7;
        int * const r= &n; // A const pointer to an int
    }

    {
        const int n = 7;
        const int * const r= &n; // A const pointer to a const int
    }

    {
        char c;                                 //                                          char
        const char Cc                  = 'a';   //                                    const char
        char *pc;                               //                         pointer to       char
        const char *pCc;                        //                         pointer to const char
        char * const Cpc               = &c;    //                   const pointer to       char
        const char * const CpCc        = &Cc;   //                   const pointer to const char
        char ** p1;                             //        pointer to       pointer to       char
        const char **p2;                        //        pointer to       pointer to const char
        char * const * p3;                      //        pointer to const pointer to       char
        const char * const * p4;                //        pointer to const pointer to const char
        char ** const p5               = &pc;   //  const pointer to       pointer to       char
        const char ** const p6         = &pCc;  //  const pointer to       pointer to const char
        char * const * const p7        = &Cpc;  //  const pointer to const pointer to       char
        const char * const * const p8  = &CpCc; //  const pointer to const pointer to const char
    }

    {
        const int *p = nullptr;         // q is a pointer to a const in
        constexpr int *q = nullptr;     // q is a const pointer to int
    }

    return 0;
}
