
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

  Class template argument deduction

  Class template argument deduction (CTAD) allows the compiler to deduce template arguments from constructor arguments.
*/

void demo2();
int main() {
  std::vector v{ 1, 2, 3 }; // deduces std::vector<int>

  std::mutex mtx;
  auto lck = std::lock_guard{ mtx }; // deduces to std::lock_guard<std::mutex>

  auto p = new std::pair{ 1.0, 2.0 }; // deduces to std::pair<double, double>

  demo2();
  return 0;
}

/*
  For user-defined types, deduction guides can be used to guide the compiler how to deduce template arguments if applicable:
*/
template <typename T>
struct container {
  container(T t) {}

  template <typename Iter>
  container(Iter beg, Iter end);
};

// deduction guide
template <typename Iter>
container(Iter b, Iter e) -> container<typename std::iterator_traits<Iter>::value_type>;

void demo2()
{
  container a{ 7 }; // OK: deduces container<int>
  container<double> aa{ 7.0 };

  std::vector v{ 1.0, 2.0, 3.0 };
  //auto b = container{ v.begin(), v.end() }; // OK: deduces container<double>          !!!LINK ERROR!!!  ??????????????????????

  //container c{ 5, 6 }; // ERROR: std::iterator_traits<int>::value_type is not a type  !!!ERROR!!!
}
