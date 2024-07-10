
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
#include <filesystem>
#include <cstddef>
#include <set>
// #include <execution>
// #include <tbb/tbb.h>
// using namespace tbb::v1;
#include <random>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::clamp
  
  Clamp given value between a lower and upper bound.
*/

int main() 
{
  int i;
  i = std::clamp(42, -1, 1); // == 1
  COUT(i);
  i = std::clamp(-42, -1, 1); // == -1
  i = std::clamp(0, -1, 1); // == 0

  // `std::clamp` also accepts a custom comparator:
  i = std::clamp(0, -1, 1, std::less<>{}); // == 0
  COUT(i);
 
  return 0;
}
