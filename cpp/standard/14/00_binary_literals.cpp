
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

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md

  Binary literals provide a convenient way to represent a base-2 number. 
  
  It is possible to separate digits with '.
*/

int main()
{
  auto a = 0b110; // == 6
  auto b = 0b1111'1111; // == 255

  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;

  uint8_t c = 0b11111111;
  double d = 0b11111111'11111111'11111111'11111111'11111111'11111111'11111111'11111111;

  std::cout << "c = " << c << std::endl;
  std::cout << "d = " << d << std::endl;
  printf("size of double = %ld\n", sizeof(double));
  printf("DBL_MAX = %e\n", DBL_MAX);
  printf("DBL_MAX = %f\n", DBL_MAX);
  
  return 0;
}
