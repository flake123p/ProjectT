#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <stdfloat> // C++23
#include <string>

#include "_float.h"

#define DBG_MSG_EN ( 1 )
#if DBG_MSG_EN
#define DBG_MSG printf
#else
#define DBG_MSG(...)
#endif


/*
    f32 to f16:

        pytorch: cuda_fp16.hpp 

            __internal_float2half(), __internal_half2float()

*/

void std_float16_t_DEMO()
{
    std::float16_t f16 = 0.1F16;
    float f32;
    std::cout << f16 << std::endl;
    printf("%lu\n", sizeof(f16));

    f32 = f16;
    printf("%f\n", f32);

    f32 = 3.1;
    f16 = (std::float16_t)f32;
    std::cout << f16 << std::endl;
}

typedef struct {
    uint32_t u;
    float f;
} F32_Golden;

typedef struct {
    uint16_t u;
    float f;
} F16_Golden;

F32_Golden F32_1_to_0[] = {
#include "golden_c/F32_1_to_0.txt"
};
F32_Golden F32_N1_to_0[] = {
#include "golden_c/F32_N1_to_0.txt"
};
F32_Golden F32_1_to_INF[] = {
#include "golden_c/F32_1_to_INF.txt"
};
F32_Golden F32_N1_to_NINF[] = {
#include "golden_c/F32_N1_to_NINF.txt"
};

F16_Golden F16_1_to_0[]     = {
#include "golden_c/F16_1_to_0.txt"
};
F16_Golden F16_N1_to_0[]    = {
#include "golden_c/F16_N1_to_0.txt"
};
F16_Golden F16_1_to_INF[]   = {
#include "golden_c/F16_1_to_INF.txt"
};
F16_Golden F16_N1_to_NINF[] = {
#include "golden_c/F16_N1_to_NINF.txt"
};

template<typename Golden_T, typename Len_T, typename Func_T>
void UIntToFloat_Test_Impl (const char *name, Golden_T golden_ary, Len_T len, Func_T toFloat) {
    //auto len = ARRAY_LEN(golden_ary);func

    using golden_float_t = decltype(golden_ary->f);

    golden_float_t tempF;
    int same;

    for (decltype(len) i = 0; i < len; i++) {
        tempF = (golden_float_t)toFloat(golden_ary[i].u);
        same = (golden_ary[i].f == tempF);

        //DBG_MSG("%s, %d, [G]%.60f, [MY]%.60f, 0x%08X, %lu\n", name, same, golden_ary[i].f, tempF, fcell.u, i);

        if (!same) {
            DBG_MSG("%s, %d, [G]%.60f, [MY]%.60f, 0x%08X, %lu\n", name, same, golden_ary[i].f, tempF, golden_ary[i].u, i);
            BASIC_ASSERT_NOEXIT(same);
        }
    }
    printf("%s : Done.\n", name);
}

template<typename FloatCell_T>
float floatUint_to_float_algo_My(uint32_t fp_u) {
    static FloatCell_T cell;
    cell.u = fp_u;
    return (float)cell.Double();
}

#define UIntToFloat_Test_F32_MACRO(ary) UIntToFloat_Test_Impl(#ary, ARRAY_AND_SIZE(ary), floatUint_to_float_algo_My<Fp32Cell>)

void UIntToFloat_Test_F32 () {
    UIntToFloat_Test_F32_MACRO(F32_1_to_0);
    UIntToFloat_Test_F32_MACRO(F32_1_to_INF);
    UIntToFloat_Test_F32_MACRO(F32_N1_to_0);
    UIntToFloat_Test_F32_MACRO(F32_N1_to_NINF);
    // auto len = ARRAY_LEN(F32_1_to_NINF);

    // Fp32Cell fp32c;
    // float tempF32;
    // int same;

    // for (decltype(len) i = 0; i < len; i++) {
    //     fp32c.u = F32_1_to_NINF[i].u;
    //     tempF32 = (float)fp32c.Double();
    //     same = F32_1_to_NINF[i].f==tempF32;
    //     DBG_MSG("%d, %f, %f, 0x%08X\n", same, F32_1_to_NINF[i].f, tempF32, fp32c.u);
    //     BASIC_ASSERT(same);
    // }
}

#define UIntToFloat_Test_F16_MACRO(ary) UIntToFloat_Test_Impl(#ary, ARRAY_AND_SIZE(ary), floatUint_to_float_algo_My<Fp16Cell>)

void UIntToFloat_Test_F16 () {
    UIntToFloat_Test_F16_MACRO(F16_1_to_0);
    UIntToFloat_Test_F16_MACRO(F16_1_to_INF);
    UIntToFloat_Test_F16_MACRO(F16_N1_to_0);
    UIntToFloat_Test_F16_MACRO(F16_N1_to_NINF);
}


int main()
{
    UIntToFloat_Test_F32();
    UIntToFloat_Test_F16();

    int s = size_of_array(F16_1_to_0);
    printf("s = %d\n", s);
    return 0;
}