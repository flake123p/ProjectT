#include "_arTen.h"
#include "mnist_for_c.h"
#include "nn.h"
#include "_lib.h"
#include "nn_mnist.h"


/*
    10 : best 10 last
    10 : best 10 duplicate for compare
    50 : best x 50
    90 : left 9 x 10
*/
#define BEST_CHILDREN_NUM 30
#define TOP_9_CHILDREN_NUM 10

class modelX mx[10 + 50 + 90 + 10];

void load_pretrain()
{
    mx[0]._L1a.unpickle_array("pt/file_0003048_1a.txt");
    mx[0]._L2w.unpickle_array("pt/file_0003048_2w.txt");
    mx[0]._L2b.unpickle_array("pt/file_0003048_2b.txt");

    for (int i = 1; i < 10; i++) {
        mx[i] = mx[0];

        mx[i].RandomBasic(0.2f, 0.2f, 0.1f);
    }
}

void gen_next_generation()
{
    // Duplicate, 10 ~ 20
    for (int i = 10; i < 20; i++) {
        mx[i] = mx[i-10];
    }

    // best x 50
    for (int i = 20; i < 45; i++) {
        mx[i] = mx[0];
        mx[i].RandomBasic(0.2f, 0.2f, 0.1f);
    }
    for (int i = 45; i < 70; i++) {
        mx[i] = mx[0];
        mx[i].RandomBasic();
    }

    // left 9 x 10
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = 70 + (i * 10) + j;
            int src = i + 1;

            mx[idx] = mx[src];
            mx[idx].RandomBasic();
        }
    }
}

void run_generation_init(ArTen<float> &images, int len, int *labels)
{
    for (int i = 10; i < 20; i++) {
        mx[i].InferBatch_HoodSum(images, len, labels, 0, len);
    }
}

void run_generation(ArTen<float> &images, int len, int *labels)
{
    for (int i = 20; i < 160; i++) {
        mx[i].InferBatch_HoodSum(images, len, labels, 0, len);
    }
}

void top_10_selection()
{
    int matchMax = 0;
    int bestIdx;
    for (int i = 10; i < 160; i++) {
        if (mx[i].matches > matchMax) {
            matchMax = mx[i].matches;
            bestIdx = i;
        }
    }
    mx[0] = mx[bestIdx];

    for (int j = 1; j < 10; j++) {
        for (int i = 10; i < 160; i++) {


        }
    }
}

void ModelX_Run() {
    PRLOC
    load_mnist();
    PRLOC

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

    load_pretrain();
    run_generation_init(trainImages, NUM_TRAIN, train_label);
    gen_next_generation();

    int epochs = 999999999;
    for (int e = 0; e < epochs; e++) {
        run_generation(trainImages, NUM_TRAIN, train_label);

        top_10_selection();

        gen_next_generation();
    }
}