
#include "all.hpp"

//
// https://en.cppreference.com/w/cpp/utility/forward
//

struct A
{
    A(int&& n) { std::cout << "A rvalue overload, n=" << n << '\n'; }
    A(int& n)  { std::cout << "A lvalue overload, n=" << n << '\n'; }
    //A(int n)   { std::cout << "A ...... overload, n=" << n << '\n'; } //error: call of overloaded ‘A(int)’ is ambiguous
};

struct B
{
    B(int&& n) { std::cout << "B rvalue overload, n=" << n << '\n'; }
};

struct C
{
    C(int n) { std::cout << "C ...... overload, n=" << n << '\n'; }
};

int main()
{
    struct A a(3);

    int x = 4;
    struct A aa(x);

    struct B b(std::forward<int>(5));

    x = 6;
    struct B bb(std::forward<int>(x));

    struct B bbb(7);
    x = 8;
    //struct B bbbb(x); //error: cannot bind rvalue reference of type ‘int&&’ to lvalue of type ‘int’

    struct C c(9);
    x = 10;
    struct C cc(x);

    struct C ccc(std::forward<int>(11));
    x = 12;
    struct C cxcc(std::forward<int>(x));

    return 0;
}
