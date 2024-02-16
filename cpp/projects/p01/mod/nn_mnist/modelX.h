#pragma once

#include "nn.h"
#include "_lib.h"

class modelX
{
private:
    /* data */
    float *buf1 = nullptr;
    float *buf2 = nullptr;
public:
    ArTen<float> _L1a;
    ArTen<float> _L2w;
    ArTen<float> _L2b;
    float sum = 0.0f;
    int matches = 0;
    // uint32_t seqL = 0;
    // uint32_t seqR = 1;

    modelX(/* args */) {
        float *buf1 = (float *)malloc(200 * sizeof(float));
        float *buf2 = (float *)malloc(200 * sizeof(float));

        _L1a.Reset({200, 784});
        _L2w.Reset({10, 200});
        _L2b.Reset({10});

        buf1 = buf1;
        buf2 = buf2;
    };
    ~modelX() {
        free_safely(buf1);
        free_safely(buf2);
    };
    // Copy assignment operator.
    modelX& operator=(const modelX& other) {
        // this->val = other.val;
        this->_L1a.copy_array(other._L1a);
        this->_L2w.copy_array(other._L2w);
        this->_L2b.copy_array(other._L2b);
        this->sum = other.sum;
        this->matches = other.matches;
        return *this;
    }

    void Init() {
        _L1a.travers_array([&](int idx, float *inst) {
            idx = idx;
            *inst = RandFloat0to1<float>();
        });
        _L2w.travers_array([&](int idx, float *inst) {
            idx = idx;
            *inst = RandFloat0to1<float>() * 2.0f - 1.0f;
        });
        _L2b.travers_array([&](int idx, float *inst) {
            idx = idx;
            *inst = RandFloat0to1<float>() * 2.0f - 1.0f;
        });
    }

    void RandomBasic(float l1a = 0.5f, float l2w = 0.5f, float l2b = 0.2f) {
        _L1a.random(l1a, 50, [&](int idx, float *inst) {
            idx = idx;
            *inst = RandFloat0to1<float>();
        });
        _L2w.random(l2w, 2, [&](int idx, float *inst) {
            idx = idx;
            *inst = RandFloat0to1<float>() * 10.0f - 5.0f;
        });
        _L2b.random(l2b, 1, [&](int idx, float *inst) {
            idx = idx;
            *inst = RandFloat0to1<float>() * 10.0f - 5.0f;
        });

        // seqR++;
    }

    float InferBatch_HoodSum (
        ArTen<float> &images, int len, int *labels,
        int start, int end,
        int *matches = nullptr
    )
    {
        // if (seqL != seqR) {
        //     seqL = seqR;
        // } else {
        //     if (matches != nullptr) {
        //         *matches = this->matches;
        //     }
        //     return this->sum;
        // }
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

        this->sum = hood_sum;
        this->matches = matched_ctr;

        return hood_sum;
    }

    void OutputFile(int index, int indents = 9) {
        const char *prefix = "pt_temp/file_";
        std::string save;

        save = prefix;
        StringAppendInt(save, index, indents);
        save = save + "_1a.txt";
        _L1a.pickle_array(save.c_str());

        save = prefix;
        StringAppendInt(save, index, indents);
        save = save + "_2w.txt";
        _L2w.pickle_array(save.c_str());

        save = prefix;
        StringAppendInt(save, index, indents);
        save = save + "_2b.txt";
        _L2b.pickle_array(save.c_str());
    }
};


