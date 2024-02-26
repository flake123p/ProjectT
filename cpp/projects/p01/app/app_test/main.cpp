#include "mnist_for_c.h"
#include "app_test.h"
#include "_float.h"

int main()
{
    //mnist_for_c_example();

    //dump_float_limits();
    //Float_Test();

    //Float_Test_ShowBF16();

    //Float_Test_Converter();

    //Float_Test_Manual();
    //Float_Test_Case00();

    Rand_Probability_Test2();
    //Stat_Test();

    // {
    //     Bf16Cell bf16;
    //     Fp16Cell fp16;

    //     bf16.u = 0x9B60;
    //     fp16.u = 0x9B60;

    //     printf("bf16 %.10f, %e\n", bf16.Double(), bf16.Double());
    //     printf("fp16 %.10f, %e\n", fp16.Double(), fp16.Double());
    // }

    return 0;
}