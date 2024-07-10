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

template <typename A1>
A1 mytemp(A1 a1)
{
    return a1 * a1;
}

template <typename A1>
A1 mytemp_dump(A1 a1)
{
    COUT(typeid(A1).name());
    COUT(sizeof(A1));
    return a1 * a1;
}

int main()
{
    {
        int a = 4;
        float b = 1.414;

        COUT(mytemp(a));
        COUT(mytemp(b));
        COUT(mytemp<double>(b));

        printf("DUMP:\n");
        mytemp_dump(a);
        mytemp_dump(b);
        mytemp_dump<double>(b);
    }

    return 0;
}