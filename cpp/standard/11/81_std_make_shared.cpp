
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

  std::make_shared is the recommended way to create instances of std::shared_ptrs due to the following reasons:

    Avoid having to use the new operator.
    Prevents code repetition when specifying the underlying type the pointer shall hold.
    It provides exception-safety. 

  --  
  Suppose we were calling a function foo like so:

    foo(std::shared_ptr<T>{new T{}}, function_that_throws(), std::shared_ptr<T>{new T{}});
  
  The compiler is free to call new T{}, then function_that_throws(), and so on... 
  
  Since we have allocated data on the heap in the first construction of a T, we have introduced a leak here. 
  
  With std::make_shared, we are given exception-safety:

    foo(std::make_shared<T>(), function_that_throws(), std::make_shared<T>());

  Prevents having to do two allocations. 
  
  When calling std::shared_ptr{ new T{} }, we have to allocate memory for T, 
  then in the shared pointer we have to allocate memory for the control block within the pointer.

  See the section on smart pointers for more information on std::unique_ptr and std::shared_ptr.
*/

int main(int argc, char *argv[])
{
    auto x = std::make_shared<int>();
    *x = 1233;
    
    auto y = x;               //  !!!OK... not unique...!!!
    COUT(*y);

    return 0;
}

