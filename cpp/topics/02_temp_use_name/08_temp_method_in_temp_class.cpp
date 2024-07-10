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
struct ReduceOp {
    T1 *src;
    T1 *dst;
    ReduceOp(T1 *inDst, T1 *inSrc): dst(inDst), src(inSrc) {};

    template <int ouput_vec_size>
    void mean() const {
        dst[0] = 0;
        for (int i = 0; i < ouput_vec_size; i++) {
            dst[0] += src[i];
        }
        dst[0] /= ouput_vec_size;
    }
};

int main()
{
    float out[1] = {0};
    float in[3] = {1, 2, 7};

    ReduceOp<float> op(out, in);

    // 1st usage:
    op.template mean<3>();

    // 2nd usage:
    // op.mean<3>();

    COUT(out[0]);

    return 0;
}