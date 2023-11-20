
#include <iostream>
#include <ostream>
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
#include <variant>
#include <optional>
#include <any>
#include <filesystem>
#include <cstddef>
#include <set>
#include <random>
#include <iterator>
#include <charconv>
#include <cassert>

#include "FunctionTraits.h"

int *abc(int a, int b, int c)
{
    return nullptr;
}

int main()
{
    {
        typedef function_traits<decltype(abc)> traits;
        static_assert(std::is_same<int *, traits::result_type>::value, "err");
        static_assert(std::is_same<int, traits::arg<0>::type>::value, "err");
        static_assert(std::is_same<int, traits::arg<1>::type>::value, "err");
        static_assert(std::is_same<int, traits::arg<2>::type>::value, "err");

        int arity = function_traits<decltype(abc)>::arity;
        assert (arity == 3);
        printf("arity = %d\n", arity);
    }
    {
        auto lambdaA = [](int i) { return long(i*10); };

        printf("A = %ld\n", lambdaA(10));

        typedef function_traits<decltype(lambdaA)> traits;
        static_assert(std::is_same<long, traits::result_type>::value, "err");
        static_assert(std::is_same<int, traits::arg<0>::type>::value, "err");

        traits::result_type a;
        traits::arg<0>::type b;
        printf("%lu %lu\n", sizeof(a), sizeof(b));

        assert(sizeof(a) == sizeof(long));
        assert(sizeof(b) == sizeof(int));
    }
    {
        auto lambdaA = []() { return 10; };
        int arity = function_traits<decltype(lambdaA)>::arity;
        //printf("arity = %d\n", arity);
        assert (arity == 0);
    }
    {
        auto lambdaA = [](int i) { return long(i*10); };
        int arity = function_traits<decltype(lambdaA)>::arity;
        //printf("arity = %d\n", arity);
        assert (arity == 1);
    }
    {
        auto lambdaA = [](int i, int j) { return i*j; };
        int arity = function_traits<decltype(lambdaA)>::arity;
        //printf("arity = %d\n", arity);
        assert (arity == 2); 
    }
    {
        auto lambdaA = [](int i, int j, int k) { return i*j*k; };
        int arity = function_traits<decltype(lambdaA)>::arity;
        //printf("arity = %d\n", arity);
        assert (arity == 3);
    }

    return 0;
}