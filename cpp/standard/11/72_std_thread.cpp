
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

    The std::thread library provides a standard way to control threads, such as spawning and killing them. 
    
    In the example below, multiple threads are spawned to do different calculations and then the program waits for all of them to finish.
*/

void foo(bool clause) { 
  for (int i = 0; i < 100000; i++) {
    if (i % 10000 == 0) {
      printf("%d\n", i);
    }
  }
}

int main()
{
  std::vector<std::thread> threadsVector;
  threadsVector.emplace_back([]() {
    // Lambda function that will be invoked
  });
  threadsVector.emplace_back(foo, true);  // thread will run foo(true)
  threadsVector.emplace_back(foo, true);  // thread will run foo(true)
  for (auto& thread : threadsVector) {
    thread.join(); // Wait for threads to finish
  }

  return 0;
}
