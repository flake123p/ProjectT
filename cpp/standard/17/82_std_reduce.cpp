
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

  std::reduce

  Fold over a given range of elements. 
  
  Conceptually similar to std::accumulate, but std::reduce will perform the fold in parallel. 
  
  Due to the fold being done in parallel, if you specify a binary operation, it is required 
  to be associative and commutative. 
  
  A given binary operation also should not change any element or invalidate any iterators within the given range.
*/

int main() 
{
  // The default binary operation is std::plus with an initial value of 0.
  const std::array<int, 4> a{ 1, 2, 3, 4 };
  {
    auto result = std::reduce(std::cbegin(a), std::cend(a)); // == 10
    CDUMP(result);
  }
  // Using a custom binary op:
  {
    auto result = std::reduce(std::cbegin(a), std::cend(a), 1, std::multiplies<>{}); // == 24
    CDUMP(result);
  }

  //
  // Additionally you can specify transformations for reducers:
  //
  {
    const auto times_ten = [](const auto a) { return a * 10; };
    auto result = std::transform_reduce(std::cbegin(a), std::cend(a), 0, std::plus<>{}, times_ten); // == 100
    CDUMP(result);
  }
  {
    const std::array<int, 4> b{ 1, 2, 3, 4 };
    const auto product_times_ten = [](const auto a, const auto b) { return a * b * 10; };

    auto result = std::transform_reduce(std::cbegin(a), std::cend(a), std::cbegin(b), 0, std::plus<>{}, product_times_ten); // == 300 == 10+40+90+160
    CDUMP(result);
  }
  
 
  return 0;
}
