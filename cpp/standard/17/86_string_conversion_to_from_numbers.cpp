
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
#include <charconv>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  String conversion to/from numbers :: #include <charconv>

  Convert integrals and floats to a string or vice-versa. 
  
  Conversions are non-throwing, do not allocate, and are more secure than the equivalents from the C standard library.

  Users are responsible for allocating enough storage required for std::to_chars, 
    or the function will fail by setting the error code object in its return value.
  
  These functions allow you to optionally pass a base (defaults to base-10) or a format specifier for floating type input.

  std::to_chars 
    returns a (non-const) char pointer which is one-past-the-end of the string 
      that the function wrote to inside the given buffer, and an error code object.

  std::from_chars 
    returns a const char pointer which on success is equal to the end pointer 
      passed to the function, and an error code object.
  
  Both error code objects returned from these functions are equal 
    to the default-initialized error code object on success.
*/

int main() 
{
  // Convert the number 123 to a std::string:
  {
    const int n = 123;

    // Can use any container, string, array, etc.
    std::string str;
    str.resize(3); // hold enough storage for each digit of `n`

    const auto [ ptr, ec ] = std::to_chars(str.data(), str.data() + str.size(), n);

    if (ec == std::errc{}) { 
      //std::cout << str << std::endl; 
      CDUMP(str);
    } // 123
    else { 
      /* handle failure */ 
    }
  }

  //Convert from a std::string with value "123" to an integer:
  {
    const std::string str{ "123" };
    int n;

    const auto [ ptr, ec ] = std::from_chars(str.data(), str.data() + str.size(), n);

    if (ec == std::errc{}) { 
      //std::cout << n << std::endl; 
      CDUMP(n);
    } // 123
    else {
      /* handle failure */ 
    }
  }

  return 0;
}
