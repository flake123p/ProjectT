
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  Selection statements with initializer

  New versions of the if and switch statements which simplify common code patterns and help users keep scopes tight.

  {
    std::lock_guard<std::mutex> lk(mx);
    if (v.empty()) v.push_back(val);
  }
  // vs.
  if (std::lock_guard<std::mutex> lk(mx); v.empty()) {
    v.push_back(val);
  }

  Foo gadget(args);
  switch (auto s = gadget.status()) {
    case OK: gadget.zip(); break;
    case Bad: throw BadFoo(s.message());
  }
  // vs.
  switch (Foo gadget(args); auto s = gadget.status()) {
    case OK: gadget.zip(); break;
    case Bad: throw BadFoo(s.message());
  }
*/
int main() {
  int x;

  if (x = 3; false) {
    COUT("if condition == true");
    COUT(x);
  } else if (x++; false) {
    COUT("elif condition == true");
  } else {
    x++;
  }

  COUT(x);

  switch (x *= 10; x) {
    default:
      COUT(x);
      break;
  }

  return 0;
}
