
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

  Nested namespaces:

  Using the namespace resolution operator to create nested namespace definitions.
*/
namespace A {
  namespace B {
    namespace C {
      int i;
    }
  }
}

// The code above can be written like this:
namespace A::B::C {
  int j;
}

int main() {
  A::B::C::i = 3;
  COUT(A::B::C::i);

  A::B::C::j = 33;
  COUT(A::B::C::j);

  return 0;
}
