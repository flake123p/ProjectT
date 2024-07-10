
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

*/

void f(int& x) {printf("lvalue ref\n");}
void f(int&& x) {printf("rvalue ref\n");}

int main()
{
    int x = 3; // `x` is an lvalue of type `int`
    int& xl = x; // `xl` is an lvalue of type `int&`
    //int&& xr = x; // compiler error -- `x` is an lvalue
    int&& xr2 = 5; // `xr2` is an lvalue of type `int&&` -- binds to the rvalue temporary, `5`

    printf("before:\n");
    printf("x   = %d\n", x);
    printf("xl  = %d\n", xl);
    printf("xr2 = %d\n", xr2);

    xl = 4;
    xr2 = 6;
    printf("after:\n");
    printf("x   = %d\n", x);
    printf("xl  = %d\n", xl);
    printf("xr2 = %d\n", xr2);

    f(x);              // calls f(int&)
    f(xl);             // calls f(int&)
    f(3);              // calls f(int&&)
    f(std::move(x));   // calls f(int&&)
    f(std::move(3));   // calls f(int&&)

    f(xr2);            // calls f(int&)
    f(std::move(xr2)); // calls f(int&&)

    return 0;
}
