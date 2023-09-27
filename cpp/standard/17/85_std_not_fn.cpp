
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
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::not_fn

  Utility function that returns the negation of the result of the given function.
*/

int main() 
{
  const std::ostream_iterator<int> ostream_it{ std::cout, " " };
  const auto is_even = [](const auto n) { return n % 2 == 0; };
  std::vector<int> v{ 0, 1, 2, 3, 4 };

  // Print all even numbers.
  std::copy_if(std::cbegin(v), std::cend(v), ostream_it, is_even); // 0 2 4
  std::cout << "\n";
  // Print all odd (not even) numbers.
  std::copy_if(std::cbegin(v), std::cend(v), ostream_it, std::not_fn(is_even)); // 1 3
  std::cout << "\n";
  return 0;
}
