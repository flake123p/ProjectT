
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Syntactic sugar for iterating over a container's elements.
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
  {
    int i = 0;
    std::array<int, 5> a {1, 2, 3, 4, 5};
    for (int& x : a) {
      x *= 2; // a == { 2, 4, 6, 8, 10 }
      printf("%d ", a[i]);
      i++;
    }
    printf("\n");
  }

  // Note the difference when using int as opposed to int&:
  {
    int i = 0;
    std::array<int, 5> a {1, 2, 3, 4, 5};
    for (int x : a) {
      x *= 2; // a == { 2, 4, 6, 8, 10 }
      printf("%d ", a[i]);
      i++;
    }
    printf("\n");
  }
  return 0;
}
