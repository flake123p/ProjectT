
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

  std::async runs the given function either asynchronously or lazily-evaluated, 
  then returns a std::future which holds the result of that function call.

  The first parameter is the policy which can be:

    std::launch::async | std::launch::deferred 
      It is up to the implementation whether to perform asynchronous execution or lazy evaluation.
    std::launch::async Run the callable object on a new thread.
    std::launch::deferred Perform lazy evaluation on the current thread.

*/

int foo() {
  /* Do something here, then return the result. */
  return 1000;
}

int main()
{
  auto handle = std::async(std::launch::async, foo);  // create an async task
  auto result = handle.get();  // wait for the result

  std::cout << "result = " << result << std::endl;
  return 0;
}
