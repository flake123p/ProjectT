
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

#include <type_traits>
/*
    Key point: Add "typename" if:

        error: dependent-name ‘...’ is parsed as a non-type, 
        but instantiation yields a type
*/

template<typename T>
T bar() {
    return (T)1.0 / (T)10.0;
}

template<typename T, typename T2>
void Bar(T &from, T2 &to)
{
    // *** NOTICE: must use typename or compile error ... ***
    to = bar<typename std::remove_reference<decltype(from())>::type>();
}

template<typename T>
class Foo {
public:
    T scalar;
    T &operator()(void) {
        return scalar;
    }
};



int main() 
{
    float a;

    static_assert(std::is_same<float, decltype(a)>::value, "wat");
    printf("a is type: float (%d)\n", std::is_same<float, decltype(a)>::value);
    
    float &b = a;
    static_assert(std::is_same<float&, decltype(b)>::value, "wat");
    printf("b is type: float& (%d)\n", std::is_same<float&, decltype(b)>::value);

    std::remove_reference<decltype(b)>::type remove_ref;
    static_assert(std::is_same<float, decltype(remove_ref)>::value, "wat");
    printf("remove_ref is type: float (%d)\n", std::is_same<float, decltype(remove_ref)>::value);

    float c = bar<std::remove_reference<decltype(b)>::type>();
    printf("c = %f\n", c);

    Foo<float> foo;
    foo.scalar = 0.2;

    float d;
    Bar(foo, d);  //                  *** NOTICE ***
    printf("d = %f\n", d);

    return 0;
}
