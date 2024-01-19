#pragma once

#include "basic.h"
#include <math.h> // pow()

/*
    The limits in float.h.

    Ref: https://www.tutorialspoint.com/c_standard_library/float_h.htm

    FLT_MAX, FLT_MIN, ...
*/
#include "float.h"
inline void dump_float_limits () {
    printf("The maximum value of float = %.10e\n", FLT_MAX);
    printf("The minimum value of float = %.10e\n", FLT_MIN);
    printf("The number of digits in the number = %.d\n", FLT_MANT_DIG);
    printf("The maximum value of double = %.10e\n", DBL_MAX);
    printf("The minimum value of double = %.10e\n", DBL_MIN);
    printf("The number of digits in the number = %.d\n", DBL_MANT_DIG);
};

/*
    IEEE754 Format:

    Ref: 
        https://en.wikipedia.org/wiki/IEEE_754
        https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407

            Alias       Total Bits  Sign Bit    Exp Bits        Fraction Bits
    BF16                16          1           8               7
    TF32                19          1           8               10
    FP16    half        16          1           5               10
    FP32    float       32          1           8               23
    FP64    double      64          1           11              52
    
*/

template<typename TypeF, typename TypeU, int ExpoBits, int FracBits>
struct FloatCellLt {
    union {
        TypeF f;
        TypeU u;
        struct {
            TypeU frac : FracBits;
            TypeU expo : ExpoBits;
            TypeU sign : 1;
        };
    };
};

template<typename TypeF, typename TypeU, int ExpoBits, int FracBits>
struct FloatCell : public FloatCellLt<TypeF, TypeU, ExpoBits, FracBits> {
    // union {
    //     TypeF f;
    //     TypeU u;
    //     struct {
    //         TypeU frac : FracBits;
    //         TypeU expo : ExpoBits;
    //         TypeU sign : 1;
    //     };
    // };
    const int expoCmpl = (1 << (ExpoBits-1)) - 1;
    const int expoBits = ExpoBits;
    const int fracBits = FracBits;

    void dump_hex(int newLine = 1) {
        printf("[%-25s] ", __func__);
        if (sizeof(TypeU) == 1)
            printf("0x%02X", (unsigned int)this->u);
        else if (sizeof(TypeU) == 2)
            printf("0x%04X", (unsigned int)this->u);
        else if (sizeof(TypeU) == 4)
            printf("0x%08X", (unsigned int)this->u);
        else if (sizeof(TypeU) == 8)
            printf("0x%016lX", (long unsigned int)this->u);
        else {
            BASIC_ASSERT(0);
        }
        
        if (newLine)
            printf("\n");
    };

    void dump_hex_fields(int newLine = 1) {
        printf("[%-25s] ", __func__);
        printf("0x%lX, ", (long unsigned int)this->sign);
        printf("0x%lX, ", (long unsigned int)this->expo);
        printf("0x%lX, ", (long unsigned int)this->frac);
        
        if (newLine)
            printf("\n");
    };

    void dump_fields(int newLine = 1) {
        int expoReal = ((int)this->expo) - expoCmpl;
        printf("[%-25s] ", __func__);
        printf("%d, ", (int)this->sign);
        printf("%d, ", expoReal);
        printf("0x%lX, ", (long unsigned int)this->frac);
        
        if (newLine)
            printf("\n");
    };

    void dump_fields_FloatFrac(int newLine = 1) {
        int expoReal = ((int)this->expo) - expoCmpl;
        printf("[%-25s] ", __func__);
        printf("%d, ", (int)this->sign);
        printf("%d, ", expoReal);
        //printf("0x%lX, ", (long unsigned int)frac);
        {
            double accu = 1.0;
            double curDbl = 1.0/2.0;
            TypeU curU = (TypeU)this->frac;
            TypeU MSB = (TypeU)1 << (FracBits - 1);

            for (int i = 0; i<FracBits; i++) {
                if (MSB & curU) {
                    accu += curDbl;
                }
                curDbl /= 2.0;
                curU <<= 1;
            }
            double exp = pow(2, expoReal);
            printf("%f * %f = %f", accu, exp, accu*exp);
        }
        
        if (newLine)
            printf("\n");
    };

    double ConvertToDouble() {
        int expoReal = ((int)this->expo) - expoCmpl;
        double accu = 1.0;
        double curDbl = 1.0/2.0;
        TypeU curU = (TypeU)this->frac;
        TypeU MSB = (TypeU)1 << (FracBits - 1);

        // Special
        if (this->expo == 0) {
            accu = 0;
            expoReal = -126;
            if (this->frac == 0) {
                if (this->sign)
                    return -0.0;
                else
                    return 0.0;
            }
        }

        for (int i = 0; i<FracBits; i++) {
            if (MSB & curU) {
                accu += curDbl;
            }
            curDbl /= 2.0;
            curU <<= 1;
        }
        double exp = pow(2, expoReal);
        //printf("%f * %f = %f", accu, exp, accu*exp);
        if (this->sign)
            return -accu*exp;
        else
            return accu*exp;
    }

    double Double() {
        return ConvertToDouble();
    }
};

using Bf16Cell = FloatCell<uint16_t, uint16_t, 8, 7>;
using Fp16Cell = FloatCell<uint16_t, uint16_t, 5, 10>;
using Fp32Cell = FloatCell<float, uint32_t, 8, 23>;
using Tf32Cell = FloatCell<float, uint32_t, 8, 10>;  // need test the bit-fields
using Fp64Cell = FloatCell<double, uint64_t, 11, 52>;

using Bf16CellLt = FloatCellLt<uint16_t, uint16_t, 8, 7>;
using Fp16CellLt = FloatCellLt<uint16_t, uint16_t, 5, 10>;
using Fp32CellLt = FloatCellLt<float, uint32_t, 8, 23>;
using Tf32CellLt = FloatCellLt<float, uint32_t, 8, 10>;  // need test the bit-fields
using Fp64CellLt = FloatCellLt<double, uint64_t, 11, 52>;

template<typename T1, typename T2>
void FloatCellConverter(T1 &dst, T2 &src) {
    dst.sign = src.sign;
    dst.expo = ((int)src.expo) - src.expoCmpl + dst.expoCmpl;
    if (dst.fracBits >= src.fracBits)
        dst.frac = ((decltype(dst.u))src.frac)<<(dst.fracBits - src.fracBits);
    else
        dst.frac = decltype(dst.u)(src.frac >> (src.fracBits - dst.fracBits));
};