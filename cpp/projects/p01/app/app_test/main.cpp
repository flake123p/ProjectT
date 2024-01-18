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
    printf("Ori=%f, F32=%f, F16=%f, BF16=%f\n\n", x, f32.Double(), f16.Double(), bf16.Double());

    float y = -56.77;
    printf("FP32 to FP64 & FP16 & BF16\n");
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
    printf("Ori=%f, F64=%f, F32=%f, BF16=%f\n\n", f16.Double(), f64.Double(), f32.Double(), bf16.Double());

    y = 0.4;
    printf("BF16 to FP64 & FP32 & FP16\n");
    f32.f = y;
    FloatCellConverter(bf16, f32);

    FloatCellConverter(f16, bf16);
    FloatCellConverter(f32, bf16);
    FloatCellConverter(f64, bf16);
    printf("Ori=%f, F64=%f, F32=%f, F16=%f\n\n", bf16.Double(), f64.Double(), f32.Double(), f16.Double());
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

int main()
{
    //mnist_for_c_example();

    dump_float_limits();
    //Float_Test();

    //Float_Test_ShowBF16();

    Float_Test_Converter();
    return 0;
}