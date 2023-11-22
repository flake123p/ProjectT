// thread example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <vector>

volatile int x[100] = {0};

void foo(int idx) 
{
  for (int i = 0; i < 2000; i++) {
    for (int j = 0; j < 1000000; j++) {
        x[idx]++;
    }
  }
}

void run(int threads)
{
  std::vector<std::thread> threadsVector;
  threadsVector.emplace_back([]() {
    // Lambda function that will be invoked
  });
  for (int i = 0; i < threads; i++) {
    threadsVector.emplace_back(foo, i);
  }

  for (auto& thread : threadsVector) {
    thread.join(); // Wait for threads to finish
  }
}

int main() 
{
  if (1) {
    std::thread a(foo, 0);
    std::thread b(foo, 0);
    std::thread c(foo, 0);
    std::thread d(foo, 0);

    std::cout << "main, foo and foo now execute concurrently...\n";

    // synchronize threads:
    a.join();
    b.join();
    c.join();
    d.join();

    std::cout << "foo and foo completed.\n";
  } else {
    std::cout << "main, foo and foo now execute concurrently...\n";
    run(8);
    std::cout << "foo and foo completed.\n";
  }

  return 0;
}