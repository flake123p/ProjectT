#include <iostream>
#include <Eigen/Dense>
#include "nn.h"
#include "_arTen.h"
#include "_arTenMat.h"
#include "_basic.h"
#include "_rand.h"

void My_Matmul_Demo()
{
    Eigen::Matrix<float, 2, 3> a;
    Eigen::Matrix<float, 2, 3> b;
    Eigen::Matrix<float, 2, 2> result;
    a(0, 0) = 2;
    a(0, 1) = 4;
    a(0, 2) = 6;
    a(1, 0) = 8;
    a(1, 1) = 10;
    a(1, 2) = 12;
    b(0, 0) = 1;
    b(0, 1) = 3;
    b(0, 2) = 5;
    b(1, 0) = 7;
    b(1, 1) = 9;
    b(1, 2) = 11;

    result = a * b.transpose();
    std::cout << "result =\n" << result << std::endl;
    printf("r:%ld, c:%ld\n", result.rows(), result.cols());

    ArTen<float> A({2, 3});
    ArTen<float> B({2, 3});
    ArTen<float> C({2, 2});
    
    Mat_To_Mat(a, A);
    Mat_To_Mat(b, B);
    //A.dump(1);
    Mat_Dump(a);
    Mat_Dump(A);

    nn_MatmulLt_RowMajor(A.array_, B.array_, C.array_, 2, 3, 2, 0, 1);
    Mat_CompareLt(result, C);
    Mat_Dump(C);
    Mat_Dump(result);

    RandSeedInit();
}

void My_Matmul_RandomTest()
{
    int m = 2;
    int k = 3;
    int n = 4;
    int loop = 10;

    for (int ctr = 0; ctr < loop; ctr++) {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> a;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> b;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> c;
        a.resize(m, k);
        b.resize(n, k);
        c.resize(m, n);
        ArTen<float> A({m, k});
        ArTen<float> B({n, k});
        ArTen<float> C({m, n});

        Mat_RandFloat0to1(a);
        Mat_RandFloat0to1(b);
        Mat_To_Mat(a, A);
        Mat_To_Mat(b, B);
        
        c = a * b.transpose();
        nn_MatmulLt_RowMajor(A.array_, B.array_, C.array_, m, k, n, 0, 1);
        // Mat_Dump(a);
        // Mat_Dump(A);
        Mat_CompareLt(a, A);
        Mat_Dump(c);
        Mat_Dump(C);
    }
}

void My_Matmul_RandomTestFull()
{
    int m, k, n;
    int loop = 100000;
    int innerLoop = 5;
    for (int lo = 0; lo < loop; lo++) {
        m = RandInRange(1, 20);
        k = RandInRange(1, 20);
        n = RandInRange(1, 20);
        for (int in = 0; in < innerLoop; in++) {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> a;
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> b;
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> c;
            a.resize(m, k);
            b.resize(n, k);
            c.resize(m, n);
            ArTen<float> A({m, k});
            ArTen<float> B({n, k});
            ArTen<float> C({m, n});

            Mat_RandFloat0to1(a);
            Mat_RandFloat0to1(b);
            Mat_To_Mat(a, A);
            Mat_To_Mat(b, B);
            
            c = a * b.transpose();
            nn_MatmulLt_RowMajor(A.array_, B.array_, C.array_, m, k, n, 0, 1);
            // Mat_Dump(a);
            // Mat_Dump(A);
            Mat_CompareLt(a, A);
            // Mat_Dump(c);
            // Mat_Dump(C);
        }
    }
}

int main()
{
    //OfficalDemo_Matmul();

    //My_Matmul_Demo();
    My_Matmul_RandomTestFull();
}