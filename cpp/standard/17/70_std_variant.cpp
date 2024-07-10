
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::variant

  The class template std::variant represents a type-safe union. 
  
  An instance of std::variant at any given time holds a value of one of its alternative types 
  (it's also possible for it to be valueless).
*/

int main() {
  std::variant<int, double> v{ 12 };
  CDUMP(std::get<int>(v)); // == 12
  CDUMP(std::get<0>(v)); // == 12
  v = 12.0;
  CDUMP(std::get<double>(v)); // == 12.0
  CDUMP(std::get<1>(v)); // == 12.0

  return 0;
}
