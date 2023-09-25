
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    The noexcept specifier specifies whether a function could throw exceptions. It is an improved version of throw().

    Non-throwing functions are permitted to call potentially-throwing functions. 
    
    Whenever an exception is thrown and the search for a handler encounters the outermost block of a non-throwing function, 
    the function std::terminate is called.
*/

void func1() noexcept;        // does not throw
void func2() noexcept(true);  // does not throw
void func3() throw();         // does not throw 
void func4() noexcept(false); // may throw

void f()
{
  throw 12;
}  // potentially-throwing

void g() noexcept {
  f();          // valid, even if f throws
  throw 42;     // valid, effectively a call to std::terminate
}

int main()
{
  g();
  return 0;
}
