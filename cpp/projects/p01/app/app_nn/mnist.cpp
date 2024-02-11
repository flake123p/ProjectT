#include "_arTen.h"
#include "mnist_for_c.h"
#include "ocv_util.h"
#include "nn.h"

#include "../../data/mnist/local_b0.txt"
#include "../../data/mnist/local_b1.txt"
#include "../../data/mnist/local_b2.txt"
#include "../../data/mnist/local_b3.txt"
#include "../../data/mnist/local_w0.txt"
#include "../../data/mnist/local_w1.txt"
#include "../../data/mnist/local_w2.txt"
#include "../../data/mnist/local_w3.txt"
/*
loss:  126.92801
loss:  19.239277
loss:  6.3718195
loss:  12.566687
loss:  9.149878
loss:  14.420612
loss:  15.936183
loss:  3.7931373
loss:  0.3377211
loss:  7.5486393
loss:  3.2628403
loss:  0.275259
loss:  7.8895726
loss:  4.5271087
loss:  0.0
[Test]  Success rate:  0.9539
[Train] Success rate:  0.97687274
*/
/*
Demo_mnist_inference_all()
    Matches = 0.953900
    Matches = 0.975367
*/
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
    printf("%f %f\n", testImages(0, 10*28+20), testImages(0, 10*28+20));

    printf("%d\n", test_label[0]);
    printf("%d\n", train_label[0]);

    nn_MatmulLt_RowMajor<float>(&testImages(0, 0), local_w0_storage, buf1, 1, 784, 200, 0, 1);
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

void MnistInferAll(ArTen<float> &images, int len, int *labels)
{
    int matchCtr = 0;
    int max;
    float *buf1 = (float *)malloc(200 * sizeof(float));
    float *buf2 = (float *)malloc(200 * sizeof(float));

    for (int i = 0; i < len; i++) {
        nn_MatmulLt_RowMajor<float>(&images(i, 0), local_w0_storage, buf1, 1, 784, 200, 0, 1);
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

        max = nn_argMax(buf2, 10);
        //printf("[%3d] max = %d, label = %d\n", i, max, labels[i]);
        if (max == labels[i]) {
            matchCtr++;
        }
    }

    printf("Matches = %f\n", ((double)matchCtr) / len);

    free(buf1);
    free(buf2);
}

void Demo_mnist_inference_all()
{
    load_mnist();

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

    // testImages.travers_array([&](int idx, float *inst) {
    //     if (idx < SIZE784) {
    //         printf("[%3d] %.50f\n", idx, *inst);
    //     }
    // });

    // printf("%f %f\n", test_image[0][10*28+20], test_image[0][10*28+21]);
    // printf("%f %f\n", testImages(0, 10*28+20), testImages(0, 10*28+20));

    // printf("%d\n", test_label[0]);
    // printf("%d\n", train_label[0]);

    MnistInferAll(testImages, NUM_TEST, test_label);
    MnistInferAll(trainImages, NUM_TRAIN, train_label);

    // MnistRunX(testImages, NUM_TEST, test_label);
}

void MnistRunX(ArTen<float> &images, int len, int *labels)
{
    int matchCtr = 0;
    int max;
    float *buf1 = (float *)malloc(200 * sizeof(float));
    float *buf2 = (float *)malloc(200 * sizeof(float));

    for (int i = 0; i < 1; i++) {
        nn_MatmulLt_RowMajorX<float>(&images(i, 0), local_w0_storage, buf1, 1, 784, 200, 0, 1);
        
        //ARRAYDUMPF(buf1, 200);

        // nn_vecAdd(buf1, local_b0_storage, buf1, 200);
        // nn_relu(buf1, buf1, 200);
        // nn_MatmulLt_RowMajor<float>(buf1, local_w1_storage, buf2, 1, 200, 200, 0, 1);
        // nn_vecAdd(buf2, local_b1_storage, buf2, 200);
        // nn_relu(buf2, buf2, 200);
        // nn_MatmulLt_RowMajor<float>(buf2, local_w2_storage, buf1, 1, 200, 200, 0, 1);
        // nn_vecAdd(buf1, local_b2_storage, buf1, 200);
        // nn_relu(buf1, buf1, 200);
        nn_MatmulLt_RowMajor<float>(buf1, local_w3_storage, buf2, 1, 200, 10, 0, 1);
        nn_vecAdd(buf2, local_b3_storage, buf2, 10);

        nn_softmax_11(buf2, buf2, 10);
        printf("labels[%d] = %d\n", i, labels[i]);
        nn_one_hot_encode(buf1, labels[i]-1, 10);
        ARRAYDUMPF(buf1, 10);


        max = nn_argMax(buf2, 10);
        //printf("[%3d] max = %d, label = %d\n", i, max, labels[i]);
        if (max == labels[i]) {
            matchCtr++;
        }
    }

    printf("Matches = %f\n", ((double)matchCtr) / len);

    free(buf1);
    free(buf2);
}

void Demo_mnist_X()
{
    load_mnist();

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

    // testImages.travers_array([&](int idx, float *inst) {
    //     if (idx < SIZE784) {
    //         printf("[%3d] %.50f\n", idx, *inst);
    //     }
    // });

    // printf("%f %f\n", test_image[0][10*28+20], test_image[0][10*28+21]);
    // printf("%f %f\n", testImages(0, 10*28+20), testImages(0, 10*28+20));

    // printf("%d\n", test_label[0]);
    // printf("%d\n", train_label[0]);

    // MnistInferAll(testImages, NUM_TEST, test_label);
    // MnistInferAll(trainImages, NUM_TRAIN, train_label);

    MnistRunX(testImages, NUM_TEST, test_label);
}