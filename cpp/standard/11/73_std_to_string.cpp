
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Converts a numeric argument to a std::string.
*/

int main()
{
  auto x = std::to_string(1.2);
  auto y = std::to_string(123);

  std::cout << x << ", len of x = " << x.length() << std::endl;
  std::cout << y << ", len of y = " << y.length() << std::endl;

  return 0;
}
