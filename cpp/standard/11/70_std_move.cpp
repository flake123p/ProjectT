
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    std::move indicates that the object passed to it may have its resources transferred. 
    
    Using objects that have been moved from should be used with care, 
    as they can be left in an unspecified state (see: What can I do with a moved-from object?).


    A definition of std::move (performing a move is nothing more than casting to an rvalue reference):

    template <typename T>
    typename remove_reference<T>::type&& move(T&& arg) {
      return static_cast<typename remove_reference<T>::type&&>(arg);
    }
*/

// Transferring std::unique_ptrs:

int main()
{
  std::unique_ptr<int> p1 {new int{0}};  // in practice, use std::make_unique
  //std::unique_ptr<int> p2 = p1; // error -- cannot copy unique pointers                  !!!ERROR!!!
  std::unique_ptr<int> p3 = std::move(p1); // move `p1` into `p3`
                                          // now unsafe to dereference object held by `p1`

  return 0;
}
