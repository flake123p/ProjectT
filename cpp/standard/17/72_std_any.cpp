
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
#include <variant>
#include <optional>
#include <any>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::any

  A type-safe container for single values of any type.
*/

int main() 
{
  std::any x {5};
  x.has_value(); // == true
  std::any_cast<int>(x); // == 5
  COUT(std::any_cast<int>(x));
  std::any_cast<int&>(x) = 10;
  std::any_cast<int>(x); // == 10
  COUT(std::any_cast<int>(x));

  return 0;
}
