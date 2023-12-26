
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

#define C10_HOST_DEVICE
#define C10_RESTRICT
#define __device__

enum class ScalarType : int8_t {
      Undefined,
  NumOptions
};

namespace c10 {
namespace detail {

template <typename T>
struct LoadImpl {
  C10_HOST_DEVICE static T apply(const void* src) {
    printf("apply1\n");
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  C10_HOST_DEVICE static bool apply(const void* src) {
    printf("apply2\n");
    static_assert(sizeof(bool) == sizeof(char), "");
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

} // namespace detail

template <typename T>
C10_HOST_DEVICE T load(const void* src) {
  printf("load1\n");
  return c10::detail::LoadImpl<T>::apply(src);
}

template <typename scalar_t>
C10_HOST_DEVICE scalar_t load(const scalar_t* src) {
  printf("load2\n");
  return c10::detail::LoadImpl<scalar_t>::apply(src);
}

} // namespace c10

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i,
            std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  printf("invoke_impl2\n");
  //std::cout << INDEX << std::endl;
  return f(c10::load<typename traits::template arg<INDEX>::type>(data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  printf("invoke2, traits::arity = %d, i = %d\n", traits::arity, i);
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  __device__ return_t operator()(arg1_t a, arg2_t b) const {
    printf("BinaryFunctor, a = %f, b = %f\n", a, b);
    return f(a, b);
  }
  BinaryFunctor(func_t f_): f(f_) {}
  private:
    func_t f;
};

template <typename T>
struct MulFunctor {
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

int main() 
{
    {// Demo1: template class
        float x = 1.1;
        float y = 9;
        float ret;
        auto f = MulFunctor<float>();
        ret = f(x, y);
        std::cout << "Demo1 ret = " << ret << std::endl;
    }
    {// Demo2: template class + template class
        auto b = BinaryFunctor<float, float, float, MulFunctor<float>>(MulFunctor<float>());
        float x = 1.1;
        float y = 9;
        float ret = b(x, y);
        std::cout << "Demo2 ret = " << ret << std::endl;
    }
    {// Demo3: invoke
        auto b = BinaryFunctor<float, float, float, MulFunctor<float>>(MulFunctor<float>());
        float x[] = {1.1, 2, 3};
        float y[] = {9, 10, 11};
        char *const C10_RESTRICT data[2] = {(char*)x, (char*)y};
        const int strides[2] = {8, 8};
        auto ret = invoke(b, data, strides, 1);
        std::cout << "Demo3 ret = " << ret << std::endl;
    }
    return 0;
}
