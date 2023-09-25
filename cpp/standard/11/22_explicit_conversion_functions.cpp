
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Conversion functions can now be made explicit using the explicit specifier.
*/

struct A {
  operator bool() const { return true; }
};

struct B {
  explicit operator bool() const { return true; }
};

int main()
{
  A a;
  if (a); // OK calls A::operator bool()
  bool ba = a; // OK copy-initialization selects A::operator bool()

  B b;
  if (b); // OK calls B::operator bool()
  //bool bb = b; // error copy-initialization does not consider B::operator bool()    !!!ERROR!!!
  return 0;
}
