
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

  __has_include
  
  __has_include (operand) operator may be used in #if and #elif expressions to check 
  whether a header or source file (operand) is available for inclusion or not.

  One use case of this would be using two libraries that work the same way, 
  using the backup/experimental one if the preferred one is not found on the system.
*/
#include <optional>

#ifdef __has_include
#  if __has_include(<optional>)
#    include <optional>
#    define have_optional 1
#  elif __has_include(<experimental/optional>)
#    include <experimental/optional>
#    define have_optional 2
#    define experimental_optional
#  else
#    define have_optional 0
#  endif
#endif

/*
  It can also be used to include headers existing under different names or locations on 
  various platforms, without knowing which platform the program is running on, OpenGL 
  headers are a good example for this which are located in OpenGL\ directory on macOS 
  and GL\ on other platforms.
*/
#ifdef __has_include
#  if __has_include(<OpenGL/gl.h>)
#    include <OpenGL/gl.h>
#    include <OpenGL/glu.h>
#  elif __has_include(<GL/gl.h>)
#    include <GL/gl.h>
#    include <GL/glu.h>
#  else
#    error No suitable OpenGL headers found.
# endif
#endif

int main() {
  COUT(have_optional);
  return 0;
}
