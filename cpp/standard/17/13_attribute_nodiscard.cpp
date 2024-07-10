
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

  [[nodiscard]] issues a warning when either a function or class has this attribute and its return value is discarded.
*/
void demo2();

[[nodiscard]] bool do_something() {
  return true; // true for success, false for failure
}

int main() {
  do_something(); // warning: ignoring return value of 'bool do_something()',
                  // declared with attribute 'nodiscard'

  demo2();
  return 0;
}

// Only issues a warning when `error_info` is returned by value.
struct [[nodiscard]] error_info {
  // ...
};

error_info do_something2() {
  error_info ei;
  // ...
  return ei;
}

void demo2()
{
  do_something2(); // warning: ignoring returned value of type 'error_info',
                   // declared with attribute 'nodiscard'
}
