#include "_arTen.h"
#include "mnist_for_c.h"
#include "nn.h"
#include "_lib.h"
#include "nn_mnist.h"
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::lock_guard
#include <stdexcept>      // std::logic_error
#include <condition_variable> // std::condition_variable

/*
    10 : best 10 last
    10 : best 10 duplicate for compare
    50 : best x 50
    90 : left 9 x 10
*/
#define BEST_CHILDREN_NUM 30
#define TOP_9_CHILDREN_NUM 10
#define TOTAL_LEN (10 + 50 + 90 + 10)

class modelX mx[TOTAL_LEN];
class Selection<int> slct(TOTAL_LEN, -1, 99999);
class Selection<float> slctf(TOTAL_LEN, -1, 99999);
int gBestMatches;
int gStuckCtr = 0;
int gEscapeCtr = 0;

void load_pretrain()
{
    mx[0]._L1a.unpickle_array("../nn_best/best_1a.txt");
    mx[0]._L2w.unpickle_array("../nn_best/best_2w.txt");
    mx[0]._L2b.unpickle_array("../nn_best/best_2b.txt");

    // mx[0] = mx[0];

    for (int i = 1; i < 10; i++) {
        mx[i] = mx[0];

        mx[i].RandomBasic(0.2f, 0.2f, 0.1f);
    }

    //
    // Link result array to model X
    //
    for (int i = 0; i < TOTAL_LEN; i++) {
        mx[i].pMatches = &(slct.val[i]);
    }
    for (int i = 0; i < TOTAL_LEN; i++) {
        mx[i].pSum = &(slctf.val[i]);
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

ArTen<float> *infer_images = nullptr;
int infer_len = 0;
int *infer_labels = nullptr;

void infer (int i) {
    mx[i].InferBatch_HoodSum(infer_images, infer_len, infer_labels, 0, infer_len);
}

void run_generation_init(ArTen<float> &images, int len, int *labels)
{
    infer_images = &images;
    infer_len = len;
    infer_labels = labels;

    std::thread threads[10];
    for (int i=0; i<10; ++i) {
        threads[i] = std::thread(infer, i);
    }
    for (auto& th : threads) {
        th.join();
    }

    for (int i = 0; i < 10; i++) {
        // mx[i].InferBatch_HoodSum(images, len, labels, 0, len);
        printf("i [%d] Matches: %d\n", i, *(mx[i].pMatches));
    }
    gBestMatches = *(mx[0].pMatches);
}

void run_generation(ArTen<float> &images, int len, int *labels)
{
    // for (int i = 20; i < 160; i++) {
    //     mx[i].InferBatch_HoodSum(images, len, labels, 0, len);
    //     printf("g [%d] Matches: %d\n", i, *(mx[i].pMatches));
    // }

    infer_images = &images;
    infer_len = len;
    infer_labels = labels;

    std::thread threads[TOTAL_LEN - 20];
    for (int i=0; i<TOTAL_LEN - 20; ++i) {
        threads[i] = std::thread(infer, i+20);
    }
    for (auto& th : threads) {
        th.join();
    }
}

void top_10_selection(int e)
{
    slct.init();

    for (int i = 0; i < 10; i++) {
        int idx = slct.max(10, TOTAL_LEN);
        mx[i] = mx[idx];
        printf("- [%d][%d] Matches: %d, Sum: %f\n", e, i, *(mx[i].pMatches), *(mx[i].pSum));
    }

    int need_update = *(mx[0].pMatches) > gBestMatches;

    printf("x [%d] 1:%6d, 2:%6d, 3:%6d, best:%d, need=%d, stuck=%d, escape=%d\n", e, *(mx[0].pMatches), *(mx[1].pMatches), *(mx[2].pMatches), gBestMatches, need_update, gStuckCtr, gEscapeCtr);

    if (need_update) {
        gStuckCtr = 0;
        gBestMatches = *(mx[0].pMatches);


        {
            std::string save;
            int indents = 9;

            save = "pt_temp/file_";
            StringAppendInt(save, e, indents);
            save = save + "_1a.txt";
            mx[0]._L1a.pickle_array(save.c_str());

            save = "pt_temp/file_";
            StringAppendInt(save, e, indents);
            save = save + "_2w.txt";
            mx[0]._L2w.pickle_array(save.c_str());

            save = "pt_temp/file_";
            StringAppendInt(save, e, indents);
            save = save + "_2b.txt";
            mx[0]._L2b.pickle_array(save.c_str());
        }
    } else {
        gStuckCtr++;
        if (gStuckCtr >= 10) {
            printf("x [%d] CHOOSE BEST SUM in TOP 10\n", e);
            
            int i = 1;
            int updateBestSum = 0;
            for (i = 1; i < 10; i++) {
                if (*(mx[i].pSum) > *(mx[0].pSum) && *(mx[i].pMatches) >= *(mx[0].pMatches) - 3) {
                    updateBestSum = 1;
                    break;
                }
            }

            printf("x [%d] CHOOSE BEST SUM in TOP 10, i = %d, updateBestSum = %d\n", e, i, updateBestSum);

            if (updateBestSum) {
                gStuckCtr = 0;
                gEscapeCtr++;
                mx[0] = mx[i];
                for (int i = 1; i < 10; i++) {
                    mx[i] = mx[0];

                    mx[i].RandomBasic(0.2f, 0.2f, 0.1f);
                }

                BASIC_ASSERT(infer_images != nullptr);
                BASIC_ASSERT(infer_len != 0);
                BASIC_ASSERT(infer_labels != nullptr);

                std::thread threads[9];
                for (int i=0; i<9; ++i) {
                    threads[i] = std::thread(infer, i+1);
                }
                for (auto& th : threads) {
                    th.join();
                }
            }
        }
    }

    // int matchMax = 0;
    // int bestIdx;
    // for (int i = 10; i < 160; i++) {
    //     if (mx[i].matches > matchMax) {
    //         matchMax = mx[i].matches;
    //         bestIdx = i;
    //     }
    // }
    // mx[0] = mx[bestIdx];

    // for (int j = 1; j < 10; j++) {
    //     for (int i = 10; i < 160; i++) {


    //     }
    // }
}

void ModelX_Run() {
    
    load_mnist();
    
    RandSeedInit();

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

        top_10_selection(e);

        gen_next_generation();
    }
}