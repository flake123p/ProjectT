#include "_arTen.h"
#include "mnist_for_c.h"
#include "ocv_util.h"
#include "nn.h"

void Demo_mnist_to_png()
{
    load_mnist();

    byte *ptr = (byte *)calloc(1*28*28, 1);

    for (int i=0; i<28*28; i++) {
        uint32_t val32 = (uint32_t)(test_image[0][i] * 255.0);
        ptr[i] = (byte)val32;
    }

    OcvUtil_BytesToImag1(ptr, 28, 28, "my.png");
}


#include "../../data/mnist/local_b0.txt"
#include "../../data/mnist/local_b1.txt"
#include "../../data/mnist/local_b2.txt"
#include "../../data/mnist/local_b3.txt"
#include "../../data/mnist/local_w0.txt"
#include "../../data/mnist/local_w1.txt"
#include "../../data/mnist/local_w2.txt"
#include "../../data/mnist/local_w3.txt"
void Demo_mnist_inference()
{
    load_mnist();

    float *buf1 = (float *)malloc(200 * sizeof(float));
    float *buf2 = (float *)malloc(200 * sizeof(float));

    ArTen<float> testImages({NUM_TEST, SIZE784});
    ArTen<float> trainImages({NUM_TRAIN, SIZE784});

    double *pTest_image = (double *)test_image;
    double *pTrain_image = (double *)train_image;

    testImages.travers_array([&](int idx, float *inst) {
        *inst = (float)pTest_image[idx];
    });
    trainImages.travers_array([&](int idx, float *inst) {
        *inst = (float)pTrain_image[idx];
    });

    printf("%f %f\n", test_image[0][10*28+20], test_image[0][10*28+21]);
    printf("%f %f\n", testImages.array_[10*28+20], testImages.array_[10*28+21]);

    printf("%d\n", test_label[0]);
    printf("%d\n", train_label[0]);

    nn_MatmulLt_RowMajor<float>(&testImages.ref({0,0}), local_w0_storage, buf1, 1, 784, 200, 0, 1);
    nn_vecAdd(buf1, local_b0_storage, buf1, 200);
    nn_relu(buf1, buf1, 200);
    nn_MatmulLt_RowMajor<float>(buf1, local_w1_storage, buf2, 1, 200, 200, 0, 1);
    nn_vecAdd(buf2, local_b1_storage, buf2, 200);
    nn_relu(buf2, buf2, 200);
    nn_MatmulLt_RowMajor<float>(buf2, local_w2_storage, buf1, 1, 200, 200, 0, 1);
    nn_vecAdd(buf1, local_b2_storage, buf1, 200);
    nn_relu(buf1, buf1, 200);
    nn_MatmulLt_RowMajor<float>(buf1, local_w3_storage, buf2, 1, 200, 10, 0, 1);
    nn_vecAdd(buf2, local_b3_storage, buf2, 10);

    int max = nn_argMax(buf2, 10);
    printf("max = %d\n", max);


    free(buf1);
    free(buf2);
}

int main()
{
    //example();
    //Demo_mnist_to_png();
    Demo_mnist_inference();
    return 0;
}