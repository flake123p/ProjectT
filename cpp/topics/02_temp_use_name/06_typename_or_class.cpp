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

int AddOpFunc(int a, int b) {
  return a + b;
}

template <class BFunc>
void mytemp_class(BFunc binop)
{
    COUT(binop(1,2));
}

template <typename BFunc>
void mytemp_typename(BFunc binop)
{
    COUT(binop(1,2));
}

int main()
{
    printf("\n--- Demo 1 --- Template with class:\n");
    mytemp_class(AddOpFunc);

    printf("\n--- Demo 2 --- Template with typename:\n");
    mytemp_typename(AddOpFunc);

    return 0;
}