#include <stdio.h>
#include <iostream>
#include <stdfloat> // C++23

int main()
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
    return 0;
}