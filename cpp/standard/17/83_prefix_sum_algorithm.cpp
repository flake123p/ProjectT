
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
// #include <execution>
// #include <tbb/tbb.h>
// using namespace tbb::v1;
#include <random>
#include <iterator>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  Prefix sum algorithms

  Support for prefix sums (both inclusive and exclusive scans) along with transformations.
*/

int main() 
{
  const std::array<int, 3> a{ 1, 2, 3 };

  std::inclusive_scan(std::cbegin(a), std::cend(a),
      std::ostream_iterator<int>{ std::cout, " " }, std::plus<>{}); // 1 3 6
  std::cout << "\n";

  std::exclusive_scan(std::cbegin(a), std::cend(a),
      std::ostream_iterator<int>{ std::cout, " " }, 0, std::plus<>{}); // 0 1 3
  std::cout << "\n";

  const auto times_ten = [](const auto n) { return n * 10; };

  std::transform_inclusive_scan(std::cbegin(a), std::cend(a),
      std::ostream_iterator<int>{ std::cout, " " }, std::plus<>{}, times_ten); // 10 30 60
  std::cout << "\n";

  std::transform_exclusive_scan(std::cbegin(a), std::cend(a),
      std::ostream_iterator<int>{ std::cout, " " }, 0, std::plus<>{}, times_ten); // 0 10 30
  
  std::cout << "\n";
  return 0;
}
