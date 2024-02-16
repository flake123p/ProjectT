#include "_arTen.h"
#include "mnist_for_c.h"
#include "nn.h"
#include "_lib.h"

float MnistRunX_HoodSum(
    ArTen<float> &images, int len, int *labels,
    int start, int end,
    float *buf1, float *buf2,
    ArTen<float> &_L1a, ArTen<float> &_L2w, ArTen<float> &_L2b,
    int *matches = nullptr
)
{
    float hood;
    float hood_sum = 0.0f;
    int matched_ctr = 0;
    int max;

    len = len;

    for (int i = start; i < end; i++) {
        nn_MatmulLt_RowMajorX<float>(&images(i, 0), _L1a.array_, buf1, 1, 784, 200, 0, 1);
        
        //ARRAYDUMPF(buf1, 200);

        // nn_vecAdd(buf1, local_b0_storage, buf1, 200);
        // nn_relu(buf1, buf1, 200);
        // nn_MatmulLt_RowMajor<float>(buf1, local_w1_storage, buf2, 1, 200, 200, 0, 1);
        // nn_vecAdd(buf2, local_b1_storage, buf2, 200);
        // nn_relu(buf2, buf2, 200);
        // nn_MatmulLt_RowMajor<float>(buf2, local_w2_storage, buf1, 1, 200, 200, 0, 1);
        // nn_vecAdd(buf1, local_b2_storage, buf1, 200);
        // nn_relu(buf1, buf1, 200);
        nn_MatmulLt_RowMajor<float>(buf1, _L2w.array_, buf2, 1, 200, 10, 0, 1);
        nn_vecAdd(buf2, _L2b.array_, buf2, 10);

        //ARRAYDUMPF(buf2, 10);
        
        nn_softmax(buf2, buf2, 10, 1.1f);
        // buf2[4] = 1;
        // ARRAYDUMPF(buf2, 10);

        // printf("labels[%d] = %d\n", i, labels[i]);
        nn_one_hot_encode(buf1, labels[i]-1, 10);
        // ARRAYDUMPF(buf1, 10);
        
        hood = nn_likely_hood_wadiff(buf2, buf1, 10, 10.0f);
        // printf(" hood = %f\n", hood);

        hood_sum += hood;

        max = nn_argMax(buf2, 10);
        //printf("[%3d] max = %d, label = %d\n", i, max, labels[i]);
        if (max == labels[i]) {
            matched_ctr++;
        }
    }

    if (matches != nullptr) {
        *matches = matched_ctr;
    }

    return hood_sum;
}

void MnistRunX(ArTen<float> &images, int len, int *labels)
{
    // int matchCtr = 0;
    // int max;
    float *buf1 = (float *)malloc(200 * sizeof(float));
    float *buf2 = (float *)malloc(200 * sizeof(float));

    ArTen<float> _L1a({200, SIZE784});
    _L1a.travers_array([&](int idx, float *inst) {
        idx = idx;
        *inst = RandFloat0to1<float>();
    });
    ArTen<float> _L2w({10, 200});
    _L2w.travers_array([&](int idx, float *inst) {
        idx = idx;
        *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
    });
    ArTen<float> _L2b({10});
    _L2b.travers_array([&](int idx, float *inst) {
        idx = idx;
        *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
    });

    ArTen<float> _L1aX({200, SIZE784});
    ArTen<float> _L2wX({10, 200});
    ArTen<float> _L2bX({10});

    // std::string file1a = "file_1a_e";
    // std::string file2w = "file_2w_e";
    // std::string file2b = "file_2b_e";

    float sum, sum2;
    int matches, matches2;
    int batches = 10;
    int batch_elem = len / batches;
    int start, end;
    int epochs = 1;
    for (int e = 0; e < epochs; e++) {
        for (int b = 0; b < batches; b++) {
            _L1aX.copy_array(_L1a);
            _L2wX.copy_array(_L2w);
            _L2bX.copy_array(_L2b);

            _L1aX.random(4, [&](int idx, float *inst) {
                idx = idx;
                *inst = RandFloat0to1<float>();
            });
            _L2wX.random(2, [&](int idx, float *inst) {
                idx = idx;
                *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
            });
            _L2bX.random(0.1, 1, [&](int idx, float *inst) {
                idx = idx;
                *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
            });

            start = b * batch_elem;
            end = start + batch_elem;
            sum = MnistRunX_HoodSum(
                images, len, labels,
                start, end,
                buf1, buf2,
                _L1a, _L2w, _L2b, 
                &matches
            );
            printf("  [e:%2d, b:%2d] sum = %f\n", e, b, sum);
            printf("  matches = %d\n", matches);

            sum2 = MnistRunX_HoodSum(
                images, len, labels,
                start, end,
                buf1, buf2,
                _L1aX, _L2wX, _L2bX,
                &matches2
            );
            printf("X [e:%2d, b:%2d] sum = %f\n", e, b, sum2);
            printf("X matches = %d\n", matches2);

            int update = 0;
            if (sum2 > sum) {
                update = 1;
            }
            if (matches2 > matches) {
                update = 1;
            }
            if (update) {
                _L1a.copy_array(_L1aX);
                _L2w.copy_array(_L2wX);
                _L2b.copy_array(_L2bX);
            }
        }
        std::string save;

        save = "file_";
        StringAppendInt(save, e, 4);
        save = save + "_1a.txt";
        _L1a.pickle_array(save.c_str());

        save = "file_";
        StringAppendInt(save, e, 4);
        save = save + "_2w.txt";
        _L2w.pickle_array(save.c_str());

        save = "file_";
        StringAppendInt(save, e, 4);
        save = save + "_2b.txt";
        _L2b.pickle_array(save.c_str());
    }

    free(buf1);
    free(buf2);
}

void MnistRunX_Lt(ArTen<float> &images, int len, int *labels)
{
    // int matchCtr = 0;
    // int max;
    float *buf1 = (float *)malloc(200 * sizeof(float));
    float *buf2 = (float *)malloc(200 * sizeof(float));

    ArTen<float> _L1a({200, SIZE784});
    _L1a.travers_array([&](int idx, float *inst) {
        idx = idx;
        *inst = RandFloat0to1<float>();
    });
    ArTen<float> _L2w({10, 200});
    _L2w.travers_array([&](int idx, float *inst) {
        idx = idx;
        *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
    });
    ArTen<float> _L2b({10});
    _L2b.travers_array([&](int idx, float *inst) {
        idx = idx;
        *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
    });

    ArTen<float> _L1aX({200, SIZE784});
    ArTen<float> _L2wX({10, 200});
    ArTen<float> _L2bX({10});

    // std::string file1a = "file_1a_e";
    // std::string file2w = "file_2w_e";
    // std::string file2b = "file_2b_e";

    float sum;//, sum2;
    int matches;//, matches2;
    // int batches = 10;
    // int batch_elem = len / batches;
    int start, end;
    int epochs = 1;
    for (int e = 0; e < epochs; e++) {
        for (int b = 0; b < 1; b++) {
            // _L1aX.copy_array(_L1a);
            // _L2wX.copy_array(_L2w);
            // _L2bX.copy_array(_L2b);

            // _L1aX.random(4, [&](int idx, float *inst) {
            //     idx = idx;
            //     *inst = RandFloat0to1<float>();
            // });
            // _L2wX.random(2, [&](int idx, float *inst) {
            //     idx = idx;
            //     *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
            // });
            // _L2bX.random(0.1, 1, [&](int idx, float *inst) {
            //     idx = idx;
            //     *inst = RandFloat0to1<float>() * 4.0f - 2.0f;
            // });

            // start = b * batch_elem;
            // end = start + batch_elem;
            start = 0;
            end = 1;
            sum = MnistRunX_HoodSum(
                images, len, labels,
                start, end,
                buf1, buf2,
                _L1a, _L2w, _L2b, 
                &matches
            );
            printf("  [e:%2d, b:%2d] sum = %f\n", e, b, sum);
            printf("  matches = %d\n", matches);

            // sum2 = MnistRunX_HoodSum(
            //     images, len, labels,
            //     start, end,
            //     buf1, buf2,
            //     _L1aX, _L2wX, _L2bX,
            //     &matches2
            // );
            // printf("X [e:%2d, b:%2d] sum = %f\n", e, b, sum2);
            // printf("X matches = %d\n", matches2);

            // int update = 0;
            // if (sum2 > sum) {
            //     update = 1;
            // }
            // if (matches2 > matches) {
            //     update = 1;
            // }
            // if (update) {
            //     _L1a.copy_array(_L1aX);
            //     _L2w.copy_array(_L2wX);
            //     _L2b.copy_array(_L2bX);
            // }
        }
        // std::string save;

        // save = "file_";
        // StringAppendInt(save, e, 4);
        // save = save + "_1a.txt";
        // _L1a.pickle_array(save.c_str());

        // save = "file_";
        // StringAppendInt(save, e, 4);
        // save = save + "_2w.txt";
        // _L2w.pickle_array(save.c_str());

        // save = "file_";
        // StringAppendInt(save, e, 4);
        // save = save + "_2b.txt";
        // _L2b.pickle_array(save.c_str());
    }

    free(buf1);
    free(buf2);
}

void MnistRunX_Pretraind(ArTen<float> &images, int len, int *labels)
{
    // int matchCtr = 0;
    // int max;
    float *buf1 = (float *)malloc(200 * sizeof(float));
    float *buf2 = (float *)malloc(200 * sizeof(float));

    ArTen<float> _L1a({200, SIZE784});
    ArTen<float> _L2w({10, 200});
    ArTen<float> _L2b({10});

    _L1a.unpickle_array("pt/file_0003236_1a.txt");
    _L2w.unpickle_array("pt/file_0003236_2w.txt");
    _L2b.unpickle_array("pt/file_0003236_2b.txt");


    ArTen<float> _L1aX({200, SIZE784});
    ArTen<float> _L2wX({10, 200});
    ArTen<float> _L2bX({10});

    // std::string file1a = "file_1a_e";
    // std::string file2w = "file_2w_e";
    // std::string file2b = "file_2b_e";

    float sum = 0, sum2 = 0;
    int matches, matches2;
    int batches = 1;
    int batch_elem = len / batches;
    int start, end;
    int epochs = 9999999;

    uint32_t old_seq = 0;
    uint32_t cur_seq = 1;

    RandSeedInit();

    for (int e = 0; e < epochs; e++) {
        int is_updated = 0;
        int matches_total = 0;
        for (int b = 0; b < batches; b++) {
            if (1) {
                _L1aX.copy_array(_L1a);
                _L2wX.copy_array(_L2w);
                _L2bX.copy_array(_L2b);
                _L1aX.random(0.5f, 50, [&](int idx, float *inst) {
                    idx = idx;
                    *inst = RandFloat0to1<float>();
                });
                _L2wX.random(0.5f, 2, [&](int idx, float *inst) {
                    idx = idx;
                    *inst = RandFloat0to1<float>() * 10.0f - 5.0f;
                });
                _L2bX.random(0.2f, 1, [&](int idx, float *inst) {
                    idx = idx;
                    *inst = RandFloat0to1<float>() * 10.0f - 5.0f;
                });
            } else {
                // _L1aX.travers_array([&](int idx, float *inst) {
                //     idx = idx;
                //     *inst = RandFloat0to1<float>();
                // });
                // _L2wX.travers_array([&](int idx, float *inst) {
                //     idx = idx;
                //     *inst = RandFloat0to1<float>() * 6.0f - 3.0f;
                // });
                // _L2bX.travers_array([&](int idx, float *inst) {
                //     idx = idx;
                //     *inst = RandFloat0to1<float>() * 6.0f - 3.0f;
                // });
            }

            start = b * batch_elem;
            end = start + batch_elem;

            if (cur_seq != old_seq) {
                sum = MnistRunX_HoodSum(
                    images, len, labels,
                    start, end,
                    buf1, buf2,
                    _L1a, _L2w, _L2b, 
                    &matches
                );
                old_seq++;
            }
            // printf("  [e:%2d, b:%2d] sum = %f\n", e, b, sum);
            // printf("  matches = %d\n", matches);

            sum2 = MnistRunX_HoodSum(
                images, len, labels,
                start, end,
                buf1, buf2,
                _L1aX, _L2wX, _L2bX,
                &matches2
            );
            printf("X [e:%2d, b:%2d] sum = %f, sum2 = %f (%d)\n", e, b, sum, sum2, sum2>sum);
            printf("X matches = %d, matches2 = %d (%d)\n", matches, matches2, matches2>matches);

            int update = 0;
            // if (sum2 > sum) {
            //     update = 1;
            // }
            if (matches2 > matches) {
                update = 1;
            }
            if (update) {
                printf("> [e:%2d, b:%2d] sum = %f (%d), matches = %d (%d)\n", 
                    e, b, sum2, sum2>sum,
                    matches2, matches2>matches);
                _L1a.copy_array(_L1aX);
                _L2w.copy_array(_L2wX);
                _L2b.copy_array(_L2bX);
                is_updated = 1;
                matches_total += matches2;
                cur_seq++;
            } else {
                matches_total += matches;
            }
        }

        if (is_updated) {
            std::string save;
            int indents = 7;

            save = "pt_temp/file_";
            StringAppendInt(save, e, indents);
            save = save + "_1a.txt";
            _L1a.pickle_array(save.c_str());

            save = "pt_temp/file_";
            StringAppendInt(save, e, indents);
            save = save + "_2w.txt";
            _L2w.pickle_array(save.c_str());

            save = "pt_temp/file_";
            StringAppendInt(save, e, indents);
            save = save + "_2b.txt";
            _L2b.pickle_array(save.c_str());
        }
        printf("- [e:%2d] matches_total = %d\n", e, matches_total);
    }

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

    //MnistRunX(testImages, NUM_TEST, test_label);
    //MnistRunX(trainImages, NUM_TRAIN, train_label);
    //MnistRunX_Lt(trainImages, NUM_TRAIN, train_label);

    MnistRunX_Pretraind(trainImages, NUM_TRAIN, train_label);
}