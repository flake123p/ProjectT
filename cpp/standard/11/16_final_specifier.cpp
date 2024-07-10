
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Specifies that a virtual function cannot be overridden in a derived class or that a class cannot be inherited from.
*/

struct A {
  virtual void foo();
};

struct B : A {
  virtual void foo() final;
};

struct C : B {
  virtual void foo(); // error -- declaration of 'foo' overrides a 'final' function
};


// Class cannot be inherited from.
struct D final {};
struct E : D {}; // error -- base 'A' is marked 'final'

int main()
{
    return 0;
}
