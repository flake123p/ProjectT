// thread example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <vector>
#include <mutex>          // std::mutex

//
// https://cplusplus.com/reference/mutex/mutex/
//
// lock()
// try_lock()
// unlock()
//
std::mutex mtx;           // mutex for critical section

volatile int x[100] = {0};

void foo() 
{
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 1000000; j++) {
        mtx.lock();
        x[0]++;
        mtx.unlock();
    }
  }
}

void bar() 
{
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 1000000; j++) {
        mtx.lock();
        x[0]--;
        mtx.unlock();
    }
  }
}

int main() 
{
  std::thread a(foo);
  std::thread b(bar);

  std::cout << "main, foo and foo now execute concurrently...\n";

  a.join();
  b.join();

  std::cout << "foo and foo completed.\n";
  printf("x[0] = %d\n", x[0]);

  return 0;
}