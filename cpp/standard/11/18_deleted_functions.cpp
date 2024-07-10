
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    A more elegant, efficient way to provide a deleted implementation of a function. Useful for preventing copies on objects.
*/

class A {
  int x;

public:
  A(int x) : x{x} {};
  //A(const A&) = delete;
  //A& operator=(const A&) = delete;
};

int main()
{
  A x {123};
  A y = x; // error -- call to deleted copy constructor
  y = x; // error -- operator= deleted
  return 0;
}
