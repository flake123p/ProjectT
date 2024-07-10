
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    std::array is a container built on top of a C-style array. 
    
    Supports common container operations such as sorting.
*/
int main()
{
  std::array<int, 3> a = {2, 1, 3};
  std::sort(a.begin(), a.end()); // a == { 1, 2, 3 }
  for (int& x : a) x *= 2; // a == { 2, 4, 6 }

  for (int x : a) printf("%d ", x);

  printf("\n");

  return 0;
}
