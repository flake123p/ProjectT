#include "_arTen.h"
#include "mnist_for_c.h"
#include "ocv_util.h"
#include "nn.h"

void Demo_mnist_to_png()
{
    load_mnist();

    u8 *ptr = (u8 *)calloc(1*28*28, 1);

    for (int i=0; i<28*28; i++) {
        uint32_t val32 = (uint32_t)(test_image[0][i] * 255.0);
        ptr[i] = (u8)val32;
    }

    OcvUtil_BytesToImag1(ptr, 28, 28, "my.png");
}

void Demo_likely_hood()
{
    float a[] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float b[] = {0.0, 0.0, 1.0, 1.0, 1.0, 1.0};

    float x[] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    float y[] = {0.0, 0.1, 0.3, 0.7, 0.9, 1.0};
    printf("x a -> %f\n", nn_likely_hood(x, a, Len(x)));
    printf("x b -> %f\n", nn_likely_hood(x, b, Len(x)));
    printf("y a -> %f\n", nn_likely_hood(y, a, Len(x)));
    printf("y b -> %f\n", nn_likely_hood(y, b, Len(x)));

    float a3[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    printf("x a3 -> %f\n", nn_likely_hood(x, a3, Len(x)));
    printf("y a3 -> %f\n", nn_likely_hood(y, a3, Len(x)));
}

int main()
{
    //example();
    //Demo_mnist_to_png();
    //Demo_mnist_inference();
    {
        // extern void Demo_mnist_inference_all();
        // Demo_mnist_inference_all();

        extern void Demo_mnist_X();
        Demo_mnist_X();

        //Demo_likely_hood();
    }
    return 0;
}