
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

    std::tie

    Creates a tuple of lvalue references. 
    
    Useful for unpacking std::pair and std::tuple objects. 
    
    Use std::ignore as a placeholder for ignored values. 
    
    In C++17, structured bindings should be used instead.
*/
int main()
{
  // With tuples...
  std::string playerName;
  std::tie(std::ignore, playerName, std::ignore) = std::make_tuple(91, "John Tavares", "NYI");

  std::cout << playerName << std::endl; 

  // With pairs...
  std::string y, n;
  std::tie(y, n) = std::make_pair("yes", "no");

  std::cout << y << ", " << n << std::endl; 

  return 0;
}
