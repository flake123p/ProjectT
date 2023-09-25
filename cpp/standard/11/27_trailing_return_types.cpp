
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    C++11 allows functions and lambdas an alternative syntax for specifying their return types.

    In C++14, decltype(auto) (C++14) can be used instead.
*/

int f() {
  return 123;
}
// vs.
auto f2() -> int {
  return 123;
}

//This feature is especially useful when certain return types cannot be resolved:
// NOTE: This does not compile!
#if 0
template <typename T, typename U>                       //              !!!ERROR!!!
decltype(a + b) add(T a, U b) {
    return a + b;
}
#endif

// Trailing return types allows this:
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

int main()
{
  {
    printf("f()  = %d\n", f());
    printf("f2() = %d\n", f2());
  }

  {
    auto g = []() -> int {
      return 123;
    };
    printf("g()  = %d\n", g());
  }

  return 0;
}
