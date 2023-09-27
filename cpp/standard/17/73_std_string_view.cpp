
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::string_view

  A non-owning reference to a string. 
  
  Useful for providing an abstraction on top of strings (e.g. for parsing).
*/

int main() 
{
  // Regular strings.
  std::string_view cppstr {"foo"};
  // Wide strings.
  std::wstring_view wcstr_v {L"baz"};
  // Character arrays.
  char array[3] = {'b', 'a', 'r'};
  std::string_view array_v(array, std::size(array));
  COUT(array_v);

  std::string str {"   trim me"};
  std::string_view v {str};
  v.remove_prefix(std::min(v.find_first_not_of(" "), v.size()));
  str; //  == "   trim me"
  v; // == "trim me"
  COUT(v);

  return 0;
}
