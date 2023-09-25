
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

    Tuples are a fixed-size collection of heterogeneous values. 
    
    Access the elements of a std::tuple by unpacking using std::tie, or using std::get.
*/
int main()
{
  // `playerProfile` has type `std::tuple<int, const char*, const char*>`.
  auto playerProfile = std::make_tuple(51, "Frans Nielsen", "NYI");
  std::cout << std::get<0>(playerProfile) << std::endl; // 51
  std::cout << std::get<1>(playerProfile) << std::endl; // "Frans Nielsen"
  std::cout << std::get<2>(playerProfile) << std::endl; // "NYI"

  return 0;
}
