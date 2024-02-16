#include "_arTen.h"
#include "mnist_for_c.h"
#include "nn.h"
#include "_lib.h"
#include "nn_mnist.h"

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

void ArTen_Pickle_Test()
{
    ArTen<float> testImages({2, 3});
    testImages.travers_array([&](int idx, float *inst) {
        *inst = (float)idx*idx;
    });
    testImages.dump(1);
    testImages.pickle_array("pickle_test_local.txt");

    ArTen<float> restore({2, 3});
    restore.unpickle_array("pickle_test_local.txt");
    restore.dump(1);

    std::string x = "abbb";
    printf("x = %s\n", x.c_str());
    StringAppendInt(x, 12, 4);
    x = x + ".txt";
    printf("x = %s\n", x.c_str());
}

void ArTen_Rand_Test()
{
    ArTen<float> testImages({2, 3});
    testImages.travers_array([&](int idx, float *inst) {
        *inst = (float)idx*idx;
    });
    testImages.dump(1);

    testImages.random(2, [&](int idx, float *inst) {
        *inst = (float)idx*100;
    });
    testImages.dump(1);

    // testImages.pickle_array("pickle_test_local.txt");

    // ArTen<float> restore({2, 3});
    // restore.unpickle_array("pickle_test_local.txt");
    // restore.dump(1);

    // std::string x = "abbb";
    // printf("x = %s\n", x.c_str());
    // StringAppendInt(x, 12, 4);
    // x = x + ".txt";
    // printf("x = %s\n", x.c_str());
}

void ArTen_X_Test()
{
    ArTen<float> a({3});
    a(0) = 0.3;
    a(1) = 0.4;
    a(2) = 0.5;
    a.dump(1);

    ArTen<float> b({3});
    b(0) = 0.3;
    b(1) = 0.59;
    b(2) = 0.69;
    b.dump(1);

    ArTen<float> c({1});

    nn_MatmulLt_RowMajorX<float>(a.array_, b.array_, c.array_, 1, 3, 1, 0, 1);
    c.dump(1);

}

void modelX_test()
{
    PRLOC
}

int main()
{
    //example();

    //ArTen_Pickle_Test();

    //ArTen_Rand_Test();

    //ArTen_X_Test();

    modelX_test();

    return 0;
}