//
// Template argument deduction
//      https://en.cppreference.com/w/cpp/language/template_argument_deduction
//
// https://stackoverflow.com/questions/10872730/can-a-template-function-be-called-with-missing-template-parameters-in-c
//


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

template <typename T>
struct AddOp {
  T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }

  static T run(T const &lhs, T const &rhs) {
    return (lhs + rhs + 10);
  }
};

int AddOpFunc(int a, int b) {
  return a + b;
}

template <class BFunc>
//template <typename BFunc> // This works too!!!
void mytemp(BFunc binop)
{
    COUT(binop(1,2));
}

int main()
{
    printf("\n--- Demo 1 --- Pass Function Call:\n");
    mytemp(AddOp<int>());

    printf("\n--- Demo 2 --- Pass Function:\n");
    mytemp(AddOp<int>::run);

    printf("\n--- Demo 3 --- Pass Pure Function:\n");
    mytemp(AddOpFunc);

    return 0;
}