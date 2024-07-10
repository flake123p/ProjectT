
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    A more elegant, efficient way to provide a default implementation of a function, such as a constructor.
*/

struct A {
    A() = default;
    A(int x) : x{x} {}
    int x {1};
};

// With inheritance:
struct B {
  B() : x{2} {}
  int x;
};

struct C : B {
  // Calls B::B
  C() = default;
};

int main()
{
    A a; // a.x == 1
    A a2 {123}; // a.x == 123

    printf("a.x  = %d\n", a.x);
    printf("a2.x = %d\n", a2.x);


    C c; // c.x == 1
    printf("c.x  = %d\n", c.x);
    return 0;
}
