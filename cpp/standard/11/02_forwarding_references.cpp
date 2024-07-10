
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Also known (unofficially) as universal references. A forwarding reference is 
    created with the syntax T&& where T is a template type parameter, or using auto&&. 
    
    This enables perfect forwarding: the ability to pass arguments while maintaining 
    their value category (e.g. lvalues stay as lvalues, temporaries are forwarded as rvalues).
*/

#if 0
// Since C++14 or later:
void f(auto&& t) {
  // ...
}
#endif

// Since C++11 or later:
template <typename T>
void f(T&& t) {
  // ...
}

int main()
{
    int x = 0; // `x` is an lvalue of type `int`
    auto&& al = x; // `al` is an lvalue of type `int&` -- binds to the lvalue, `x`
    auto&& ar = 0; // `ar` is an lvalue of type `int&&` -- binds to the rvalue temporary, `0`

    {
        int x = 0;
        f<int>(0); // T is int, deduces as f(int &&) => f(int&&)
        f<int&>(x); // T is int&, deduces as f(int& &&) => f(int&)

        int& y = x;
        f<int&>(y); // T is int&, deduces as f(int& &&) => f(int&)

        int&& z = 0; // NOTE: `z` is an lvalue with type `int&&`.
        f<int&>(z); // T is int&, deduces as f(int& &&) => f(int&)
        f<int>(std::move(z)); // T is int, deduces as f(int &&) => f(int&&)
    }
    return 0;
}
