
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Converting constructors will convert values of braced list syntax into constructor arguments.
*/

struct A {
  A(int) {printf("A 1 arg\n");}
  A(int, int) {printf("A 2 arg\n");}
  A(int, int, int) {printf("A 3 arg\n");}
};

// Note that the braced list syntax does not allow narrowing:

// Note that if a constructor accepts a std::initializer_list, it will be called instead:
struct B {
  B(int) {printf("B 1 arg\n");}
  B(int, int) {printf("B 2 arg\n");}
  B(int, int, int) {printf("B 3 arg\n");}
  B(std::initializer_list<int>) {printf("B initializer_list\n");}
};

int main()
{
  {
    A a {0, 0}; // calls A::A(int, int)
    A b(0, 0); // calls A::A(int, int)
    A c = {0, 0}; // calls A::A(int, int)
    A d {0, 0, 0}; // calls A::A(int, int, int)
  }

  {
    A a(1.1); // OK
    //A b {1.1}; // Error narrowing conversion from double to int     !!!ERROR!!!
  }

  {
    B a {0, 0};     // calls B::B(std::initializer_list<int>)
    B b(0, 0);      // calls B::B(int, int)
    B c = {0, 0};   // calls B::B(std::initializer_list<int>)
    B d {0, 0, 0};  // calls B::B(std::initializer_list<int>)
  }
  return 0;
}
