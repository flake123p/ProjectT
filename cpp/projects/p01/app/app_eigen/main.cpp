#include <iostream>
#include <Eigen/Dense>

/*
https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
*/
void OfficalDemo_Add_Sub()
{
    //Eigen::Matrix2d a;                // same
    Eigen::Matrix<double, 2, 2> a;      // same

    a << 1, 2,
         3, 4;
    Eigen::MatrixXd b(2,2);
    b << 2, 3,
         1, 4;
    std::cout << "a + b =\n" << a + b << std::endl;
    std::cout << "a - b =\n" << a - b << std::endl;
    std::cout << "Doing a += b;" << std::endl;
    a += b;
    std::cout << "Now a =\n" << a << std::endl;
    Eigen::Vector3d v(1,2,3);
    Eigen::Vector3d w(1,0,0);
    std::cout << "-v + w - v =\n" << -v + w - v << std::endl;
}

void OfficalDemo_Matmul()
{
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;
    Eigen::Vector2d u(-1,1), v(2,0);
    std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
    std::cout << "Here is mat*u:\n" << mat*u << std::endl;
    std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
    std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
    std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
    std::cout << "Let's multiply mat by itself" << std::endl;
    mat = mat*mat;
    std::cout << "Now mat is mat:\n" << mat << std::endl;
}

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
}

int main()
{
    //OfficalDemo_Matmul();

    My_Matmul_Demo();
}