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

// x is constant expression
template <typename T, int x = 3>
void mytemp(T y = 4)
{
    COUT(x);
    COUT(y);
}

int main()
{
    printf("\n--- Demo 1 ---\n");
    mytemp<int>();

    printf("\n--- Demo 2 ---\n");
    mytemp<double>();

    printf("\n--- Demo 3 ---\n");
    mytemp<int, 33>();

    printf("\n--- Demo 4 ---\n");
    mytemp(44);

    printf("\n--- Demo 5 ---\n");
    mytemp<double, 33>(44);

    int i = 1233;
    printf("\n--- Demo 6 ---\n");
    mytemp<double, 33>(i);

    return 0;
}