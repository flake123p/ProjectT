#include "_lib.h"
#include <stdio.h>

void Rand_Probability_Test()
{
    RandSeedInit();

    int ctr[4] = {0};
    float p[] = {0.0, 7.0, 3.0, 0.0};

    printf("%f\n", SumOfArray(p));

    Multinomial mulNom(p, size_of_array(p));

    for (int i = 0; i < 100000; i++) {
        size_t idx = mulNom.Roll();
        if (idx >= 4) {
            printf("idx = %lu\n", idx);
            BASIC_ASSERT(0);
        } else {
            ctr[idx]++;
        }
    }

    for (size_t i = 0; i < size_of_array(ctr); i++) {
        printf("[%3lu] %d\n", i, ctr[i]);
    }

    mulNom.CalcOutput();
    for (size_t i = 0; i < mulNom.accuLen; i++) {
        printf("%lu, %f\n", i, mulNom.output[i]);
    }
}