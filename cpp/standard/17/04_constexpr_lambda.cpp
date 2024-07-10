
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <climits>
#include <cfloat>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

    constexpr lambda:

    Compile-time lambdas using constexpr.

*/
constexpr int addOne(int n) {
    return [n] { return n + 1; }(); // execute lambda function and return
}

int main() {
    auto identity = [](int n) constexpr { return n; };
    static_assert(identity(123) == 123);
    CDUMP(identity);
    CDUMP(identity(123));

    //
    //
    //
    constexpr auto add = [](int x, int y) {
        auto L = [=] { return x; };
        auto R = [=] { return y; };
        return [=] { return L() + R(); };  // return lambda function
    };

    static_assert(add(1, 2)() == 3);

    //
    //
    //
    static_assert(addOne(1) == 2);

    return 0;
}
