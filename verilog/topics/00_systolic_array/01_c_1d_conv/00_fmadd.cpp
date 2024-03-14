#include <immintrin.h>
#include <iostream>

int main() {
    // Initialize arrays
    float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[8] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float c[8] = {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    // Perform FMA operation
    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    __m256 vc = _mm256_loadu_ps(c);
    __m256 result = _mm256_fmadd_ps(va, vb, vc);

    // Store the result
    float res[8];
    _mm256_storeu_ps(res, result);

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}