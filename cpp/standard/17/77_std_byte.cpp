
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::byte

  The new std::byte type provides a standard way of representing data as a byte. 
  
  Benefits of using std::byte over char or unsigned char is that it is not a character type, 
  and is also not an arithmetic type; while the only operator overloads available are bitwise operations.

  Note that std::byte is simply an enum, and braced initialization of enums become possible thanks to direct-list-initialization of enums.
*/

int main() 
{
  std::byte a {0};
  std::byte b {0xFF};
  int i = std::to_integer<int>(b); // 0xFF
  COUT(i);
  std::byte c = a & b;
  int j = std::to_integer<int>(c); // 0
  COUT(j);

  return 0;
}
