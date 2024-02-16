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
    modelX& operator=(const modelX& other)
    {
        // this->val = other.val;
        this->_L1a.copy_array(other._L1a);
        this->_L2w.copy_array(other._L2w);
        this->_L2b.copy_array(other._L2b);
        return *this;
    }
};


