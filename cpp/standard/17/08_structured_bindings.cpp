
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

  Structured bindings:
  
  A proposal for de-structuring initialization, that would allow writing auto [ x, y, z ] = expr; 
  where the type of expr was a tuple-like object, whose elements would be bound to the variables x, 
  y, and z (which this construct declares).
  
  Tuple-like objects include std::tuple, std::pair, std::array, and aggregate structures.
*/
using Coordinate = std::pair<int, int>;

Coordinate origin() {
  return Coordinate{1, 2};
}

int main() {
  const auto [ x, y ] = origin();
  
  CDUMP(x); // == 1
  CDUMP(y); // == 2

  //
  //
  //
  std::unordered_map<std::string, int> mapping {
    {"a", 1},
    {"b", 2},
    {"c", 3}
  };

  // Destructure by reference.
  for (const auto& [key, value] : mapping) {
    // Do something with key and value
    printf("%s, %d\n", key.c_str(), value);
  }
  return 0;
}
