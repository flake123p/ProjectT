
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

    New rules for auto deduction from braced-init-list
    
    Changes to auto deduction when used with the uniform initialization syntax. 
    
    Previously, auto x {3}; deduces a std::initializer_list<int>, which now deduces to int.

*/

int main() {
    //auto x1 {1, 2, 3}; // error: not a single element                  !!!ERROR!!!
    auto x2 = {1, 2, 3}; // x2 is std::initializer_list<int>
    auto x3 {3}; // x3 is int
    auto x4 {3.0}; // x4 is double

    //CDUMP(x1);
    //CDUMP(x2);
    for (auto e : x2) {
        CDUMP(e);
    }
    printf("\n");

    CDUMP(x3);
    CDUMP(x4);

    return 0;
}
