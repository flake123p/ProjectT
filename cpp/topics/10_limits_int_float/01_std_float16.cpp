#include <stdio.h>
#include <iostream>
#include <stdfloat> // C++23
#include <_float.h>

void Demo_std_float16() {
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

void Demo_std_float16_compare0() {
#define SRC1 0.333984
#define SRC2 127.4375

    double d0 = SRC1;
    double d1 = SRC2;
    double dr = d0 + d1;
    float f0 = SRC1;
    float f1 = SRC2;
    float fr = f0 + f1;
    std::float16_t ff0 = (std::float16_t)SRC1;
    std::float16_t ff1 = (std::float16_t)SRC2;
    std::float16_t ffr = ff0 + ff1;

    printf("%.10f, %.10f, %.10f <- double add\n", d0, d1, dr);
    printf("%.10f, %.10f, %.10f <- float add\n", f0, f1, fr);

    std::cout << ff0 << ", " << ff1 << ", " << ffr << " <- std::float16_t add" << std::endl;

    Fp16Cell f16a, f16b, f16r;
    Fp32Cell f32;
    f32.f = f0;
    FloatCellConverter(f16a, f32);
    f32.f = f1;
    FloatCellConverter(f16b, f32);

    float tempF32 = f16u_to_f32(f16a.u) + f16u_to_f32(f16b.u);
    f32.f = tempF32;
    FloatCellConverter(f16r, f32);
    printf("%.10f, %.10f, %.10f, %.10f <- My F16 Add\n", f16a.Double(), f16b.Double(), tempF32, f16r.Double());
#undef SRC1
#undef SRC2
}

void Demo_std_float16_compare1() {
#define SRC1 2.9609375000
#define SRC2 6.9023437500

    double d0 = SRC1;
    double d1 = SRC2;
    double dr = d0 + d1;
    float f0 = SRC1;
    float f1 = SRC2;
    float fr = f0 + f1;
    std::float16_t ff0 = (std::float16_t)SRC1;
    std::float16_t ff1 = (std::float16_t)SRC2;
    std::float16_t ffr = ff0 + ff1;

    printf("%.10f, %.10f, %.10f <- double add\n", d0, d1, dr);
    printf("%.10f, %.10f, %.10f <- float add\n", f0, f1, fr);

    std::cout << ff0 << ", " << ff1 << ", " << ffr << " <- std::float16_t add" << std::endl;

    Fp16Cell f16a, f16b, f16r;
    Fp32Cell f32;
    f32.f = f0;
    FloatCellConverter(f16a, f32);
    f32.f = f1;
    FloatCellConverter(f16b, f32);

    float tempF32 = f16u_to_f32(f16a.u) + f16u_to_f32(f16b.u);
    f32.f = tempF32;
    FloatCellConverter(f16r, f32);
    printf("%.10f, %.10f, %.10f, %.10f <- My F16 Add\n", f16a.Double(), f16b.Double(), tempF32, f16r.Double());

    tempF32 = 9.8671875;
    printf("tempF32 = %.10f,  0x%04X\n", tempF32, f32_to_f16u(tempF32));
    tempF32 = 9.859375;
    printf("tempF32 = %.10f,  0x%04X\n", tempF32, f32_to_f16u(tempF32));
#undef SRC1
#undef SRC2
}

int main()
{
    Demo_std_float16_compare1();
    return 0;
}