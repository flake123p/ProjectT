
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  [[fallthrough]], [[nodiscard]], [[maybe_unused]] attributes

  C++17 introduces three new attributes: [[fallthrough]], [[nodiscard]] and [[maybe_unused]].

  [[fallthrough]] indicates to the compiler that falling through in a switch statement is intended behavior. 
  This attribute may only be used in a switch statement, and must be placed before the next case/default label.
*/

int main() {
  int n = 1;

  switch (n) {
    case 1: 
      // ...
      COUT(1);
      [[fallthrough]];
    case 2:
      // ...
      //break;
      COUT(2);
    case 3:
      // ...
      COUT(3);
      [[fallthrough]];
    default:
      // ...
      COUT(9999);
  }

  return 0;
}
