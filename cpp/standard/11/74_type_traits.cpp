
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Type traits defines a compile-time template-based interface to query or modify the properties of types.
*/

int main()
{
  static_assert(std::is_integral<int>::value);
  static_assert(std::is_same<int, int>::value);
  static_assert(std::is_same<std::conditional<true, int, double>::type, int>::value);

  return 0;
}
