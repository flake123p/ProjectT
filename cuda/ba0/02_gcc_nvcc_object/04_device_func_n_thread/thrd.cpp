#include <stdio.h>
#include "hpc.h"
#include <thread>
#include <vector>

void foo(EUnit *units, int idx) 
{
    units = units += idx;
    units->run();
}

void runThreads(EUnit *units, int num)
{
  std::vector<std::thread> threadsVector;
  for (int i = 0; i < num; i++) {
    threadsVector.emplace_back(foo, units, i);
  }

  for (auto& thread : threadsVector) {
    thread.join(); // Wait for threads to finish
  }
}
