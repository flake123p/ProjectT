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


int main()
{
    //mnist_for_c_example();

    dump_float_limits();
    Float_Test();
    return 0;
}