
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
#include <cassert>

using namespace std;

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
    C++ 14
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md#return-type-deduction

    New user-defined literals for standard library types, including new built-in literals for chrono and basic_string. 
    
    These can be constexpr meaning they can be used at compile-time. 
    
    Some uses for these literals include compile-time integer parsing, binary literals, and imaginary number literals.
*/

int main(int argc, char *argv[])
{
    using namespace std::chrono_literals;

    auto day = 24h;

    COUT(day.count()); // == 24

    COUT(std::chrono::duration_cast<std::chrono::minutes>(day).count()); // == 1440

    return 0;
}
