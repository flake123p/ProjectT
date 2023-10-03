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

template<typename T1, typename T2, typename T3>
struct madd {
    T1 operator()(T1 in1, T2 in2, T3 in3) const {
    return in1 * in2 + in3;
  }
};

template <typename scalar_t, template <typename, typename, typename> class Epilogue>
void vec_madd(scalar_t *output, const scalar_t *input, int count)
{
    Epilogue<scalar_t, scalar_t, scalar_t> epilogue;

    for (int i = 0; i < count; i++) {
        output[i] = epilogue(input[i], 2, 1);
    }
}

int main()
{
    float out[3] = {0};
    float in[3] = {1, 2, 3};

    vec_madd<float, madd>(out, in, 3);   // !!!NOTICE!!!

    COUT(out[0]);
    COUT(out[1]);
    COUT(out[2]);

    return 0;
}