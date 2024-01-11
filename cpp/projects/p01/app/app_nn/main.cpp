#include "mnist_for_c.h"
#include "ocv_util.h"

void demo_mnist_to_png()
{
    load_mnist();

    byte *ptr = (byte *)calloc(1*28*28, 1);

    for (int i=0; i<28*28; i++) {
        uint32_t val32 = (uint32_t)(test_image[0][i] * 255.0);
        ptr[i] = (byte)val32;
    }

    OcvUtil_BytesToImag1(ptr, 28, 28, "my.png");
}

int main()
{
    //example();
    demo_mnist_to_png();
    return 0;
}