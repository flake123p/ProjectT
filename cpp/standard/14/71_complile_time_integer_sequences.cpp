
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

    The class template std::integer_sequence represents a compile-time sequence of integers. 
    
    There are a few helpers built on top:
        std::make_integer_sequence<T, N> - creates a sequence of 0, ..., N - 1 with type T.
        std::index_sequence_for<T...> - converts a template parameter pack into an integer sequence.

*/

// Convert an array into a tuple:
template<typename Array, std::size_t... I>
decltype(auto) a2t_impl(const Array& a, std::integer_sequence<std::size_t, I...>) {
  return std::make_tuple(a[I]...);
}

template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
decltype(auto) a2t(const std::array<T, N>& a) {
  return a2t_impl(a, Indices());
}

int main(int argc, char *argv[])
{
    std::array<int, 3> a = {2, 1, 3};
    //std::sort(a.begin(), a.end()); // for test

    auto b = a2t(a);

    std::cout << std::get<0>(b) << std::endl; // 2
    std::cout << std::get<1>(b) << std::endl; // 1
    std::cout << std::get<2>(b) << std::endl; // 3

    return 0;
}
