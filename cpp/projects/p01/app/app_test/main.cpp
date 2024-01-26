#include "mnist_for_c.h"
#include "_float.h"

void Float_Test() 
{
    Fp32Cell f32;
    float x = 0.33;

    f32.f = x;
    f32.dump_hex();

    Fp64Cell f64;
    double y = 33.0f;

    f64.f = y;
    f64.dump_hex();

    printf("f32\n");
    f32.dump_hex_fields();
    f32.dump_fields();
    f32.dump_fields_FloatFrac();
    printf("f64\n");
    f64.dump_hex_fields();
    f64.dump_fields();
    f64.dump_fields_FloatFrac();

    printf("f32 to f64 manual\n");
    f64.sign = f32.sign;
    f64.expo = ((int)f32.expo) - f32.expoCmpl + f64.expoCmpl;
    f64.frac = ((uint64_t)f32.frac)<<29;
    f64.dump_fields();
    f64.dump_fields_FloatFrac();

    printf("f32 to f64 auto\n");
    f64.f = 0; //clear
    f64.f = f32.f;
    f64.dump_fields();
    f64.dump_fields_FloatFrac();
}

void Float_Test_Converter() 
{
    Bf16Cell bf16;
    Fp16Cell f16;
    Fp32Cell f32;
    Fp64Cell f64;

    double x = 1.3;
    printf("FP64 to FP32 & FP16 & BF16\n");
    f64.f = x;
    FloatCellConverter(bf16, f64);
    FloatCellConverter(f16, f64);
    FloatCellConverter(f32, f64);
    printf("Ori=%f, F32=%f, F16=%f, BF16=%f\n", x, f32.Double(), f16.Double(), bf16.Double());
    x = 0;
    f64.f = x;
    FloatCellConverter(bf16, f64);
    FloatCellConverter(f16, f64);
    FloatCellConverter(f32, f64);
    printf("Ori=%f, F32=%f, F16=%f, BF16=%f\n", x, f32.Double(), f16.Double(), bf16.Double());
    x = -0;
    f64.f = x;
    FloatCellConverter(bf16, f64);
    FloatCellConverter(f16, f64);
    FloatCellConverter(f32, f64);
    printf("Ori=%f, F32=%f, F16=%f, BF16=%f\n\n", x, f32.Double(), f16.Double(), bf16.Double());

    float y = -56.77;
    printf("FP32 to FP64 & FP16 & BF16\n");
    f32.f = y;
    FloatCellConverter(bf16, f32);
    FloatCellConverter(f16, f32);
    FloatCellConverter(f64, f32);
    printf("Ori=%f, F64=%f, F16=%f, BF16=%f\n", y, f64.Double(), f16.Double(), bf16.Double());
    y = 0;
    f32.f = y;
    FloatCellConverter(bf16, f32);
    FloatCellConverter(f16, f32);
    FloatCellConverter(f64, f32);
    printf("Ori=%f, F64=%f, F16=%f, BF16=%f\n", y, f64.Double(), f16.Double(), bf16.Double());
    y = -0;
    f32.f = y;
    FloatCellConverter(bf16, f32);
    FloatCellConverter(f16, f32);
    FloatCellConverter(f64, f32);
    printf("Ori=%f, F64=%f, F16=%f, BF16=%f\n\n", y, f64.Double(), f16.Double(), bf16.Double());

    y = -0.333;
    printf("FP16 to FP64 & FP32 & BF16\n");
    f32.f = y;
    FloatCellConverter(f16, f32);
    FloatCellConverter(bf16, f16);
    FloatCellConverter(f32, f16);
    FloatCellConverter(f64, f16);
    printf("Ori=%f, F64=%f, F32=%f, BF16=%f\n", f16.Double(), f64.Double(), f32.Double(), bf16.Double());
    y = 0;
    f32.f = y;
    FloatCellConverter(f16, f32);
    FloatCellConverter(bf16, f16);
    FloatCellConverter(f32, f16);
    FloatCellConverter(f64, f16);
    printf("Ori=%f, F64=%f, F32=%f, BF16=%f\n", f16.Double(), f64.Double(), f32.Double(), bf16.Double());
    y = -0;
    f32.f = y;
    FloatCellConverter(f16, f32);
    FloatCellConverter(bf16, f16);
    FloatCellConverter(f32, f16);
    FloatCellConverter(f64, f16);
    printf("Ori=%f, F64=%f, F32=%f, BF16=%f\n\n", f16.Double(), f64.Double(), f32.Double(), bf16.Double());

    y = 0.4;
    printf("BF16 to FP64 & FP32 & FP16\n");
    f32.f = y;
    FloatCellConverter(bf16, f32);
    FloatCellConverter(f16, bf16);
    FloatCellConverter(f32, bf16);
    FloatCellConverter(f64, bf16);
    printf("Ori=%f, F64=%f, F32=%f, F16=%f\n", bf16.Double(), f64.Double(), f32.Double(), f16.Double());
    y = 0;
    f32.f = y;
    FloatCellConverter(bf16, f32);
    FloatCellConverter(f16, bf16);
    FloatCellConverter(f32, bf16);
    FloatCellConverter(f64, bf16);
    printf("Ori=%f, F64=%f, F32=%f, F16=%f\n", bf16.Double(), f64.Double(), f32.Double(), f16.Double());
    y = -0;
    f32.f = y;
    FloatCellConverter(bf16, f32);
    FloatCellConverter(f16, bf16);
    FloatCellConverter(f32, bf16);
    FloatCellConverter(f64, bf16);
    printf("Ori=%f, F64=%f, F32=%f, F16=%f\n\n", bf16.Double(), f64.Double(), f32.Double(), f16.Double());

    {
        float y = -56.77;
        float result;
        Fp16Cell f16;
        f16.u = f32_to_f16(y);
        result = f16_to_f32(f16.u);

        printf("Ori=%f, toF16=%f, toF16toF32=%f\n\n", y, f16.Double(), result);
    }
}

void Float_Test_ShowBF16()
{
    /*
        From Pytorch & Py754

        1.3     1.296875    0x3FA6      0x1.4c00000000000p+0
        200.7   201.0       0x4349      0x1.9200000000000p+7
        -20.4   -20.375     0xC1A3     -0x1.4600000000000p+4
    */

    Bf16Cell bf16;
    bf16.u = 0x3FA6;
    printf("bf16 float is %f\n", bf16.ConvertToDouble());
    bf16.u = 0x4349;
    printf("bf16 float is %f\n", bf16.ConvertToDouble());
    bf16.u = 0xC1A3;
    printf("bf16 float is %f\n", bf16.ConvertToDouble());
}

long double Exp2(int exp) 
{
    constexpr long double two = 2.0;

    long double ret = 1.0;
    for (int e = 0; e < exp; e++) {
        ret = ret * two;
    }

    return ret;
}

void Float_Test_Manual()
{
    struct GoldenFP32 {
        uint32_t u;
        float f;
    };

    struct GoldenFP32 goldFp32[] = {
        {0x00000000, 0.0f},
        {0x80000000, -0.0f},
        {0x3f800000, 1.0f},
        {0xbf800000, -1.0f},
        {0x00000001, 1.401298e-45},
        {0x80000001, -1.401298e-45},
        {0x00400000, 5.877472e-39},
        {0x80400000, -5.877472e-39},
        {0x007fffff, 0.00000000000000000000000000000000000001175494210692},
        {0x807fffff, -1.175494e-38},
        {0x00800000, 1.175494e-38},
        {0x80800000, -1.175494e-38},
        {0x7f7fffff, 3.402823e+38},
        {0x7f7fffff, -3.402823e+38},
    };

    Fp32Cell f32;
    
    // f32.sign = 0;
    // f32.expo = 0x0;
    // f32.frac = 0x7FFFFF;
    f32.u = 0x007fffff;
    //f32.f = 1.401298e-45;

    printf("[TestMan] 0x%08X, %.50f, %e ... %e\n", f32.u, f32.f, f32.f, f32.Double());

    for (size_t i = 0; i < sizeof(goldFp32)/sizeof(goldFp32[0]); i++) {
        f32.u = goldFp32[i].u;
        printf("%ld ... %d, %.50f, %.50f\n", i, ((float)f32.Double()==goldFp32[i].f), (float)f32.Double(), goldFp32[i].f);
    }

    // printf("%Lf\n", Exp2(0));
    // printf("%Lf\n", Exp2(1));
    // printf("%Lf\n", Exp2(2));
    // printf("%Lf\n", Exp2(3));

    // long double x = (long double)1.0 + ((long double)1.0/Exp2(23));

    // long double y = (long double)1.0 / Exp2(127);

    // // printf("[TestMan] %.50Lf, %Le\n", x, x);
    // printf("[TestMan] %.50Lf, %Le\n", y, y);

    // float a, b, c, d;

    // a = 200000;
    // //b = 0.008;
    // b = 0.007; // disappear
    // c = 0.007;
    // d = 0.007;

    // printf("%.20f\n", (a + b) + (c + d));
}

void Float_Test_Case00() 
{
    Fp32Cell a32, b32;
    a32.f = 127.4375;
    b32.f = 0.534180;
    Fp16Cell a16, b16;
    FloatCellConverter(a16, a32);
    FloatCellConverter(b16, b32);

    printf("a32 = %f (0x%08X), b32 = %f\n", a32.f, a32.u, b32.f);

    printf("a16 = %f (0x%04X), b16 = %f\n", a16.Double(), a16.u, b16.Double());

    float a = f16u_to_f32(a16.u);
    float b = f16u_to_f32(b16.u);
    Fp16Cell result;
    result.u = f32_to_f16u(a + b);
    printf("Result = %f\n", result.Double());
}

int main()
{
    //mnist_for_c_example();

    //dump_float_limits();
    //Float_Test();

    //Float_Test_ShowBF16();

    //Float_Test_Converter();

    //Float_Test_Manual();
    Float_Test_Case00();
    return 0;
}