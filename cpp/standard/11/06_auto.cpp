
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
void demo2();

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    auto-typed variables are deduced by the compiler according to the type of their initializer.
*/

int main()
{
    auto a = 3.14; // double
    auto b = 1; // int
    auto& c = b; // int&
    auto d = { 0 }; // std::initializer_list<int>
    auto&& e = 1; // int&&
    auto&& f = b; // int&
    auto g = new auto(123); // int*
    const auto h = 1; // const int
    auto i = 1, j = 2, k = 3; // int, int, int

    //auto l = 1, m = true, n = 1.61; // error: inconsistent deduction -- `l` deduced to be int, `m` is bool
    // auto o; // error -- `o` requires initializer

    //
    // Extremely useful for readability, especially for complicated types:
    //
    {
        std::vector<int> v;
        {
            std::vector<int>::const_iterator cit = v.cbegin();
        }
        // v.s.
        {
            auto cit = v.cbegin();
        }
    }

    demo2();

    return 0;
}

//
// Functions can also deduce the return type using auto. 
// In C++11, a return type must be specified either explicitly, or using decltype like so:
//
template <typename X, typename Y>
auto add(X x, Y y) -> decltype(x + y) {
    return x + y;
}

void demo2()
{
    auto ret = add(1.5,1.5);
    printf("ret = %f\n", ret);
}