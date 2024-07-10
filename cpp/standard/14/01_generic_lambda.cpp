
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

/*
    C++ 14
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md#return-type-deduction

    Generic lambda expressions
    C++14 now allows the auto type-specifier in the parameter list, enabling polymorphic lambdas.
*/

int main(int argc, char *argv[])
{
    auto identity = [](auto x) { return x; }; // ********** DEMO SENSATION ********** //
    int three = identity(3); // == 3
    std::string foo = identity("foo"); // == "foo"

    std::cout << three << std::endl;
    std::cout << foo << std::endl;

    auto square = [](auto x) { return x*x; }; // ********** DEMO SENSATION ********** //
    int intA = 12;
    cout << square(intA) << endl;

    double doubleB = 0.7;
    cout << square(doubleB) << endl;
    
    return 0;
}