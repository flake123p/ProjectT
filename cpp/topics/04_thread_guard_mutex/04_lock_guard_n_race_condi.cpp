//
// https://cplusplus.com/reference/mutex/lock_guard/
//

// lock_guard example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::lock_guard
#include <stdexcept>      // std::logic_error

// Not so good because it stops other threads.
#define ENABLE_NOT_SO_GOOD_LOCK_GUARD ( 1 )

std::mutex mtx;

void print_thread_id (int id) {
#if ENABLE_NOT_SO_GOOD_LOCK_GUARD
  std::lock_guard<std::mutex> lck (mtx);
#endif
  for (int i = 0; i < 10000; i++) {
    printf("%d%d%d%d%d%d", id, id, id, id, id, id);
    printf("%d%d%d%d%d%d\n", id, id, id, id, id, id);
  }
}

int main ()
{
  std::thread threads[2];
  // spawn 10 threads:
  for (int i=0; i<2; ++i)
    threads[i] = std::thread(print_thread_id,i+1);

  for (auto& th : threads) th.join();

  return 0;
}