
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

  std::ref(val) is used to create object of type std::reference_wrapper that holds reference of val. 
  
  Used in cases when usual reference passing using & does not compile or & is dropped due to type deduction. 
  
  std::cref is similar but created reference wrapper holds a const reference to val.
*/
int main()
{
  // create a container to store reference of objects.
  auto val = 99;
  auto _ref = std::ref(val);
  _ref++;
  auto _cref = std::cref(val);
  //_cref++; does not compile
  std::vector<std::reference_wrapper<int>>vec; // vector<int&>vec does not compile
  vec.push_back(_ref); // vec.push_back(&i) does not compile
  std::cout << val << std::endl; // prints 100
  std::cout << vec[0] << std::endl; // prints 100
  std::cout << _cref << std::endl; // prints 100
  return 0;
}
