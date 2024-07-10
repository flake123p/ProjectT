#include <immintrin.h>
#include <iostream>

float conv_1d3(float *in, float *k)
{
    float c[8] = {.0f, .0f, .0f, .0f, .0f, .0f, .0f, .0f};

    __m256 va = _mm256_loadu_ps(in);
    __m256 vb = _mm256_loadu_ps(k);
    __m256 vc = _mm256_loadu_ps(c);
    __m256 result = _mm256_fmadd_ps(va, vb, vc);

    // Store the result
    float res[8];
    _mm256_storeu_ps(res, result);

    return res[0] + res[1] + res[2];
}

int main() {
    // Initialize arrays
    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float input_buf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float k[8] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 3; i++) {
        input_buf[0] = input[i + 0];
        input_buf[1] = input[i + 1];
        input_buf[2] = input[i + 2];
        float result = conv_1d3(input_buf, k);
        printf("[%d] %f\n", i, result);
    }

    return 0;
}