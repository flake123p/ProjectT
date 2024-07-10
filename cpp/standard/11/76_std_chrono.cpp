
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread> // chrono

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    The chrono library contains a set of utility functions and types that deal with durations, clocks, and time points. 
    
    One use case of this library is benchmarking code:
*/
int main()
{
  std::chrono::time_point<std::chrono::steady_clock> start, end;
  start = std::chrono::steady_clock::now();
  // Some computations...
  for (int i = 0; i<100000; i+=2) {
    i = i - 1;
  }
  end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  double t = elapsed_seconds.count(); // t number of seconds, represented as a `double`

  printf("t = %f\n", t);

  return 0;
}
