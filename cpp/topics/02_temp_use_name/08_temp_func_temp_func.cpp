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

template<typename T1>
T1 my_add (T1 in1, T1 in2){
    return in1 + in2;
};

template<typename T1>
T1 my_mul (T1 in1, T1 in2){
    return in1 * in2;
};

template <typename scalar_t, typename Func>
void vec_op(scalar_t *output, const scalar_t *input1, const scalar_t *input2, int count, Func func)
{
    for (int i = 0; i < count; i++) {
        output[i] = func(input1[i], input2[i]);
    }
}

int main()
{
    float out[3] = {0};
    float in1[3] = {1, 2, 3};
    float in2[3] = {2, 3, 4};
    //out[0] = my_add(in1[0], in2[0]);

    printf("my_add:\n");
    vec_op<float>(out, in1, in2, 3, my_add<float>);   // !!!NOTICE!!!

    COUT(out[0]);
    COUT(out[1]);
    COUT(out[2]);

    printf("my_mul:\n");
    vec_op<float>(out, in1, in2, 3, my_mul<float>);   // !!!NOTICE!!!

    COUT(out[0]);
    COUT(out[1]);
    COUT(out[2]);

    return 0;
}