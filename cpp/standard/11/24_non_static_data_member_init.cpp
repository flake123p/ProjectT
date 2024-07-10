
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Allows non-static data members to be initialized where they are declared, 
    potentially cleaning up constructors of default initializations.
*/

// Default initialization prior to C++11
class Human {
  public:
    Human() : age{10} {}
    unsigned age;
};
// Default initialization on C++11
class Human2 {
  public:
    unsigned age {20}; //                                       !!!NOTICE!!!
};

int main()
{
  {
    Human h1;
    printf("h1.age = %d\n", h1.age);
    Human2 h2;
    printf("h2.age = %d\n", h2.age);
  }

  return 0;
}
