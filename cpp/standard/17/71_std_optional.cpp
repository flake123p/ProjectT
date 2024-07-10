
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::optional

  The class template std::optional manages an optional contained value, 
  i.e. a value that may or may not be present. 
  
  A common use case for optional is the return value of a function that may fail.
*/

std::optional<std::string> create(bool b) {
  if (b) {
    return "Godzilla";
  } else {
    return {};
  }
}

int main() 
{
  create(false).value_or("empty"); // == "empty"

  create(true).value(); // == "Godzilla"

  // optional-returning factory functions are usable as conditions of while and if
  if (auto str = create(false)) {
    // ...
    printf("123\n");
  } else {
    printf("456\n");
  }

  return 0;
}
