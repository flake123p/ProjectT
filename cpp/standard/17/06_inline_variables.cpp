
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

  Inline variables

  The inline specifier can be applied to variables as well as to functions. 
  
  A variable declared inline has the same semantics as a function declared inline.

  --

  Inline Variables in C++ 17
  
    https://www.geeksforgeeks.org/cpp-17-inline-variables/
*/

// It can also be used to declare and define a static member variable, such that it does not need to be initialized in the source file.
struct S2 {
  S2() : id{count++} {}
  ~S2() { count--; }
  int id;
  static inline int count{0}; // declare and initialize count to 0 within the class
};

struct S { int x; };
inline S x1 = S{321}; // mov esi, dword ptr [x1]
                      // x1: .long 321

int main() {
  // Disassembly example using compiler explorer.
  


  S x2 = S{123};        // mov eax, dword ptr [.L_ZZ4mainE2x2]
                        // mov dword ptr [rbp - 8], eax
                        // .L_ZZ4mainE2x2: .long 123
  
  COUT(x1.x);
  COUT(x2.x);


  COUT(S2::count);
  {
    S2 a, b;
    COUT(S2::count);
  }
  COUT(S2::count);

  return 0;
}
