// thread example
#include <iostream>       // std::cout
#include <thread>         // std::thread, std::chrono
#include <vector>

volatile int x[100] = {0};

void foo(int idx) 
{
  for (int i = 0; i < 200; i++) {
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
  if (0) {
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
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    start = std::chrono::steady_clock::now();
    run(448);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double t = elapsed_seconds.count(); // t number of seconds, represented as a `double`

    printf("t = %f\n", t);
    std::cout << "foo and foo completed.\n";
  }

  return 0;
}