
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Assertions that are evaluated at compile-time.
                                   at compile-time.
                                    at compile-time.
*/

int main()
{
    constexpr int x = 0;
    constexpr int y = 1;

    static_assert(x != y, "x == y");
    static_assert(x == y, "x != y"); // error: static assertion failed: x != y

    int w = 0;
    int z = 0;
    static_assert(w == z, "w != z"); // error: non-constant condition for static assertion

    return 0;
}
