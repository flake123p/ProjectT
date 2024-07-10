
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

using namespace std;

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
    C++ 14
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md#return-type-deduction

    The decltype(auto) type-specifier also deduces a type like auto does. 
    
    However, it deduces return types while keeping their references and cv-qualifiers, while auto will not.
*/

// Note: Especially useful for generic code!

// Return type is `int`.
auto return_type_is_int(const int& i) {
    return i;
}

// Return type is `const int&`.
decltype(auto) return_type_is_const_int_r(const int& i) {
    return i;
}

int main(int argc, char *argv[])
{
    const int x = 0;
    auto x1 = x; // int
    decltype(auto) x2 = x; // const int
    int y = 0;
    int& y1 = y;
    auto y2 = y1; // int
    decltype(auto) y3 = y1; // int&
    int&& z = 0;
    auto z1 = std::move(z); // int
    decltype(auto) z2 = std::move(z); // int&&

    {
        int x = 123;
        static_assert(std::is_same<const int&, decltype(return_type_is_int(x))>::value == 0);
        static_assert(std::is_same<int, decltype(return_type_is_int(x))>::value == 1);
        static_assert(std::is_same<const int&, decltype(return_type_is_const_int_r(x))>::value == 1);
    }

    return 0;
}
