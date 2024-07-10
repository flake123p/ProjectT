
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
#include <typeinfo>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://en.cppreference.com/w/cpp/language/typeid
*/

struct Base {}; // non-polymorphic
struct Derived : Base {};
 
struct Base2 { virtual void foo() {} }; // polymorphic
struct Derived2 : Base2 {};

int main()
{
  // Non-polymorphic lvalue is a static type
  Derived d1;
  Base& b1 = d1;
  std::cout << "reference to non-polymorphic base: " << typeid(b1).name() << '\n';

  Derived2 d2;
  Base2& b2 = d2;
  std::cout << "reference to polymorphic base: " << typeid(b2).name() << '\n';
  
  return 0;
}
