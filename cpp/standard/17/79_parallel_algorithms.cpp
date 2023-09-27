
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
#include <random>
// #include <execution>
// #include <tbb/tbb.h>
// using namespace tbb::v1;

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  Parallel algorithms :: #include <execution>
  
  Many of the STL algorithms, such as the copy, find and sort methods, started to 
  support the parallel execution policies: seq, par and par_unseq which translate 
  to "sequentially", "parallel" and "parallel unsequenced".
*/

int main() 
{
  std::vector<int> longVector{2,1,3};
  // Find element using parallel execution policy
  auto result1 = std::find(std::execution::par, std::begin(longVector), std::end(longVector), 2);
  // Sort elements using sequential execution policy
  auto result2 = std::sort(std::execution::seq, std::begin(longVector), std::end(longVector));  // !!!ERROR!!! ????
 
  return 0;
}
