
#include "all.hpp"

//
//https://learn.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2010/f90831hc(v=vs.100)?redirectedfrom=MSDN
//

/*
    All variables, including nonmodifiable (const) variables, are lvalues. 

    An rvalue is a temporary value ...
*/

// lvalues_and_rvalues2.cpp
int main()
{
   int i=0, j=0, *p=&j;

   // Correct usage: the variable i is an lvalue.
   i = 7;

   // Incorrect usage: The left operand must be an lvalue (C2106).
#if 0
   7 = i; // C2106
   j * 4 = 7; // C2106
#endif

   // Correct usage: the dereferenced pointer is an lvalue.
   *p = i; 

   const int ci = 7;
   // Incorrect usage: the variable is a non-modifiable lvalue (C3892).
#if 0
   ci = 9; // C3892
#endif

   // Correct usage: the conditional operator returns an lvalue.    KEYPOINT
   ((i < 3) ? i : j) = 777;

   printf("i = %d\n", i);
   printf("j = %d\n", j);
}

/*
The examples in this topic illustrate correct and incorrect usage when operators are not overloaded. 

By overloading operators, you can make an expression such as j * 4 an lvalue.  [KEYPOINT HERE]
*/