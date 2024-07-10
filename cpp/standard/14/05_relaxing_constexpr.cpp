
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

    In C++11, constexpr function bodies could only contain a very limited set of syntaxes, 
    including (but not limited to): typedefs, usings, and a single return statement. 
    
    In C++14, the set of allowable syntaxes expands greatly to include the most common 
    syntax such as if statements, multiple returns, loops, etc.
*/

constexpr int factorial(int n) {
  if (n <= 1) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}

int main(int argc, char *argv[])
{
    const int x = factorial(5);
    
    static_assert(x == 120);

    constexpr int y = factorial(5);
    
    static_assert(y == 120);

    int z = factorial(5);
    
    assert(z == 120);

    return 0;
}
