
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

    C++14 allows variables to be templated:
*/

template<class T>
constexpr T pi = T(3.1415926535897932385);

template<class T>
constexpr T e  = T(2.7182818284590452353);

int main(int argc, char *argv[])
{
    float pi32 = pi<float>;

    COUT(pi32);

    double e64 = e<double>;

    COUT(e64);

    return 0;
}
