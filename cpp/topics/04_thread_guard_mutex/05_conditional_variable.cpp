//
// https://cplusplus.com/reference/mutex/lock_guard/
//

// lock_guard example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::lock_guard
#include <stdexcept>      // std::logic_error
#include <condition_variable> // std::condition_variable

// Not so good because it stops other threads.
#define ENABLE_NOT_SO_GOOD_LOCK_GUARD ( 1 )

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id (int id) {
    std::unique_lock<std::mutex> lck(mtx);
    while (!ready) cv.wait(lck);
    // ...
    std::cout << "thread " << id << '\n';
}

void go() {
    std::unique_lock<std::mutex> lck(mtx);
    ready = true;
    cv.notify_all();
}

int main ()
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
        threads[i] = std::thread(print_id,i);

    std::cout << "10 threads ready to race...\n";
    go();                       // go!

    for (auto& th : threads) th.join();

    return 0;
}