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

template<typename T>
struct AA {
    T val;
    AA(T in_val){val = in_val;};
};

template<typename T>
struct BB {
    T *val;
    BB(T &in_val){val = &in_val;};
};

int main()
{
    AA aa(3);
    COUT(aa.val);

    BB bb(aa); //                                                         Note: template template here

    COUT(bb.val->val);
    aa.val = 4;
    COUT(bb.val->val);

    return 0;
}