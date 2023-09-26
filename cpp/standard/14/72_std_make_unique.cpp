
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
#include <climits>
#include <cfloat>
#include <cassert>

using namespace std;

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  C++ 14
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md#return-type-deduction

  std::make_unique is the recommended way to create instances of std::unique_ptrs due to the following reasons:
    Avoid having to use the new operator.
    Prevents code repetition when specifying the underlying type the pointer shall hold.
    Most importantly, it provides exception-safety. Suppose we were calling a function foo like so:

      foo(std::unique_ptr<T>{new T{}}, function_that_throws(), std::unique_ptr<T>{new T{}});
  
  The compiler is free to call new T{}, then function_that_throws(), and so on... 
  
  Since we have allocated data on the heap in the first construction of a T, 
  we have introduced a leak here. With std::make_unique, we are given exception-safety:

    foo(std::make_unique<T>(), function_that_throws(), std::make_unique<T>());

  See the section on smart pointers (C++11) for more information on std::unique_ptr and std::shared_ptr.
*/

int main(int argc, char *argv[])
{
    auto x = std::make_unique<int>();
    *x = 1277;
    
    //auto y = x;               //  !!!ERROR... unique...!!!
    auto y = std::move(x);      //  !!!OK!!!
    COUT(*y);

    return 0;
}
