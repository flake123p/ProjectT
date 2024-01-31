#include "_lib.h"
#include <stdio.h>
#include "_stat.h"

/*
    C/C++ Lib:

        https://github.com/christianbender/statistic

        [ GSL ] https://www.gnu.org/software/gsl/

*/
void Stat_Test()
{
    using float_t = double;

    /*
        u  = 0.016822  MY
        sd = 0.129701  MY

        u  = 2.24 (Golden) https://www.youtube.com/watch?v=zeJD6dqJ5lo&ab_channel=3Blue1Brown
    
        u  = 0.02018667   R lang  https://influentialpoints.com/notes/n3rvari.htm
        sd = 0.1420798    R lang  https://influentialpoints.com/notes/n3rvari.htm  sqrt(mean((y-mean(y))^2)


        Python:
            [self]      var = 0.01682222222222222
            [self]      sd  = 0.12970050972229147
            [NUMPY]     var = 0.01682222222222222
            [NUMPY]     sd  = 0.12970050972229147
            [STAT MOD] pvar = 0.01682222222222222
            [STAT MOD]  var = 0.020186666666666665
            [STAT MOD]  psd = 0.12970050972229147
            [STAT MOD]  sd  = 0.1420797897896343
    */
    float_t ary[] = {0.41, 0.25, 0.15, 0.1, 0.06, 0.03};
    float_t sum, mean, var, sd;

    Stat_SumMeanVarianceStddev(ary, Len(ary), sum, mean, var, sd);

    // printf("%f\n", ary[0]);
    printf("sum    = %f\n", sum);
    printf("mean   = %f\n", mean);
    printf("var    = %f\n", var);
    printf("sd     = %f\n", sd);
}