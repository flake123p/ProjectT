
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Returns the arguments passed to it while maintaining their value category and cv-qualifiers. 
    
    Useful for generic code and factories.
*/

// An example of a function wrapper which just forwards other A objects to a new A object's copy or move constructor:
struct A {
  A() = default;
  A(const A& o) { std::cout << "copied" << std::endl; }
  A(A&& o) { std::cout << "moved" << std::endl; }
};

template <typename T>
A wrapper(T&& arg) {
  return A{std::forward<T>(arg)};
}

int main()
{
  wrapper(A{}); // moved
  A a;
  wrapper(a); // copied
  wrapper(std::move(a)); // moved

  return 0;
}
