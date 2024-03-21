#include <iostream>
#include <Eigen/Dense>
#include "_float.h"
#include "_rand.h"
#include "_arTen.h"
#include "nn.h"
#include <float.h>
#include <stdlib.h>
#include <thread>         // std::thread, std::chrono
#include <vector>
#include "opti00.hpp"
#include "opti01.hpp"

//
// https://github.com/flame/how-to-optimize-gemm/wiki#the-gotoblasblis-approach-to-optimizing-matrix-matrix-multiplication---step-by-step
//

template<typename Func_T>
void time_prof(Func_T func, std::string id)
{
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    start = std::chrono::steady_clock::now();
    {
        func();
    }
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double t = elapsed_seconds.count() * 1000; // t number of seconds, represented as aa `double`
    printf("t = %13.6f ms [%s]\n", t, id.c_str());
}

#define SHAPE0 1000

class ArTen<float> AA({SHAPE0, SHAPE0});
class ArTen<float> BB({SHAPE0, SHAPE0});
class ArTen<float> CC({SHAPE0, SHAPE0});

Eigen::MatrixXf aa = Eigen::MatrixXf(SHAPE0, SHAPE0);
Eigen::MatrixXf bb = Eigen::MatrixXf(SHAPE0, SHAPE0);
Eigen::MatrixXf cc = Eigen::MatrixXf(SHAPE0, SHAPE0);

void init()
{
    float f32;

    for (int i = 0; i < SHAPE0; i++) {
        for (int j = 0; j < SHAPE0; j++) {
            f32 = RandFloat0to1<float>();
            aa(i, j) = f32;
            AA(i, j) = f32;

            f32 = RandFloat0to1<float>();
            bb(i, j) = f32;
            BB(i, j) = f32;
        }
    }

    cc = aa * bb;
    
    // std::cout << "result =\n" << result << std::endl;
    // printf("r:%ld, cc:%ld\n", result.rows(), result.cols());
}

void test_nn_Matmul_accuracy() 
{
    nn_MatmulLt_RowMajor<float>(AA.array_, BB.array_, CC.array_, SHAPE0, SHAPE0, SHAPE0, 0, 0);

    float max_diff_abs = 0;
    int diff_ctr = 0;

    for (int i = 0; i < SHAPE0; i++) {
        for (int j = 0; j < SHAPE0; j++) {
            float diff = CC(i, j) - cc(i, j);
            if (diff) {
                float diff_abs = fabs(diff);

                max_diff_abs = fmax(max_diff_abs, diff_abs);
                if (diff_abs > 0.00001) {
                    //printf("i:%3d, j:%3d, diff:%10.6f\n", i, j , diff);
                    diff_ctr++;
                }
            }
        }
    }

    printf("%s() : max_diff_abs = %f, diff_ctr = %d\n", __func__, max_diff_abs, diff_ctr);
}

void test_opti00_accuracy() 
{
    memset(CC.array_, 0, SHAPE0 * SHAPE0);
    MY_MMult_Opti00<float>(SHAPE0, SHAPE0, SHAPE0, AA.array_, SHAPE0, BB.array_, SHAPE0, CC.array_, SHAPE0);

    float max_diff_abs = 0;
    int diff_ctr = 0;

    for (int i = 0; i < SHAPE0; i++) {
        for (int j = 0; j < SHAPE0; j++) {
            float diff = CC(i, j) - cc(i, j);
            if (diff) {
                float diff_abs = fabs(diff);

                max_diff_abs = fmax(max_diff_abs, diff_abs);
                if (diff_abs > 0.00001) {
                    //printf("i:%3d, j:%3d, diff:%10.6f\n", i, j , diff);
                    diff_ctr++;
                }
            }
        }
    }

    printf("%s() : max_diff_abs = %f, diff_ctr = %d\n", __func__, max_diff_abs, diff_ctr);
}

void test_nn_Matmul_time() 
{
    int loop = 1;

    time_prof (
        [&]() -> void {
            for (int i = 0; i < loop; i++) {
                cc = aa * bb;
            }
        }, 
        "eigen_mm"
    );
    time_prof (
        [&]() -> void {
            for (int i = 0; i < loop; i++) {
                nn_MatmulLt_RowMajor<float>(AA.array_, BB.array_, CC.array_, SHAPE0, SHAPE0, SHAPE0, 0, 0);
            }
        }, 
        "nn_mm"
    );
    time_prof (
        [&]() -> void {
            for (int i = 0; i < loop; i++) {
                MY_MMult_Opti00<float>(SHAPE0, SHAPE0, SHAPE0, AA.array_, SHAPE0, BB.array_, SHAPE0, CC.array_, SHAPE0);
            }
        }, 
        "Opti00"
    );
    time_prof (
        [&]() -> void {
            for (int i = 0; i < loop; i++) {
                MY_MMult_Opti01<float>(SHAPE0, SHAPE0, SHAPE0, AA.array_, SHAPE0, BB.array_, SHAPE0, CC.array_, SHAPE0);
            }
        }, 
        "Opti01"
    );
}

int main()
{

    init();

    test_nn_Matmul_accuracy();
    test_opti00_accuracy();

    test_nn_Matmul_time();
}