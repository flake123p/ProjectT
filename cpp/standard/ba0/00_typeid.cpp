
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

  const std::type_info& ti1 = typeid(A);
  const std::type_info& ti2 = typeid(A);
  
  assert(&ti1 == &ti2); // not guaranteed
  assert(ti1 == ti2); // guaranteed
  assert(ti1.hash_code() == ti2.hash_code()); // guaranteed
  assert(std::type_index(ti1) == std::type_index(ti2)); // guaranteed
*/

int main()
{
  int myint = 50;
  unsigned int *myuintptr;
  std::string mystr = "string";
  double mydouble = 3.14;
  double *mydoubleptr = nullptr;

  COUT(typeid(myint).name());
  COUT(typeid(myuintptr).name());
  COUT(typeid(*myuintptr).name());
  COUT(typeid(mystr).name());
  COUT(typeid(mydouble).name());
  COUT(typeid(mydoubleptr).name());
  COUT(typeid(*mydoubleptr).name());

  uint8_t u8;
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  COUT(typeid(u8).name());
  COUT(typeid(u16).name());
  COUT(typeid(u32).name());
  COUT(typeid(u64).name());

  long mylong = 10;
  const std::type_info& ti1 = typeid(mylong);
  COUT(ti1.name());

  long long myll;
  unsigned long long myull;
  COUT(typeid(myll).name());
  COUT(typeid(myull).name());
  
  return 0;
}
