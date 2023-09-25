
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    The copy constructor and copy assignment operator are called when copies are made, 
    and with C++11's introduction of move semantics, 
    there is now a move constructor and move assignment operator for moves.
*/

struct A {
  std::string s;
  A() : s{"test"} {printf("    1st cons\n");}
  A(const A& o) : s{o.s} {printf("    2nd cons\n");}
  A(A&& o) : s{std::move(o.s)} {printf("    move cons\n");}
  A& operator=(A&& o) {
    printf("    move as-op\n");
    s = std::move(o.s);
    return *this;
  }
};

A f(A a) {
  return a;
}

int main()
{
  printf("Line %d\n", __LINE__);
  A a1 = f(A{}); // move-constructed from rvalue temporary
  printf("Line %d, a1.s = %s\n", __LINE__, a1.s.c_str());
  A a2 = std::move(a1); // move-constructed using std::move
  printf("Line %d\n", __LINE__);
  A a3 = A{};
  printf("Line %d\n", __LINE__);
  a2 = std::move(a3); // move-assignment using std::move
  printf("Line %d\n", __LINE__);
  a1 = f(A{}); // move-assignment from rvalue temporary
  printf("Line %d\n", __LINE__);
  return 0;
}
