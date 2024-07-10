
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

    Using an auto return type in C++14, the compiler will attempt to deduce the type for you. 
    
    With lambdas, you can now deduce its return type using auto, which makes returning a deduced reference or rvalue reference possible.
*/

// Deduce return type as `int`.
auto foo(int i) {
    return i;
}

template <typename T>
auto& f(T& t) {
  return t;
}

int main(int argc, char *argv[])
{
    // Returns a reference to a deduced type.
    auto g = [](auto& x) -> auto& { return f(x); }; // ********** DEMO SENSATION ********** //
    int y = 123;
    int& z = g(y); // reference to `y`

    printf("z=%d\n", z);
    z = 999;
    printf("y=%d\n", y);

    double a = 2.3;
    double& b = g(a);
    COUT(b);
    b = 3.4;
    COUT(a);

    return 0;
}
