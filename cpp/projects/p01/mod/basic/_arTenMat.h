#pragma once

#include "_arTen.h"
#include "_rand.h"
#include <iomanip>    // std::setw and std::setfill
#include <type_traits>

//
// Eigen <-> ArTen
//
template<typename TypeFrom, typename TypeTo>
void Mat_To_Mat(TypeFrom &from, TypeTo &to)
{
    BASIC_ASSERT(from.rows() == to.rows());
    BASIC_ASSERT(from.cols() == to.cols());

    for (long r = 0; r < from.rows(); r++) {
        for (long c = 0; c < from.cols(); c++) {
            to(r, c) = from(r, c);
        }
    }
}

//
// Eigen, ArTen
//
template<typename TypeFrom>
void Mat_Dump(TypeFrom &from, bool flatten = false, int w = 9)
{
    printf("%s, r=%ld, c=%ld\n", __func__, from.rows(), from.cols());

    for (long r = 0; r < from.rows(); r++) {
        for (long c = 0; c < from.cols(); c++) {
            std::cout << std::setfill(' ') << std::setw(w) << from(r, c) << ",";
            if (flatten)
                std::cout << std::endl;
        }
        if (!flatten)
            std::cout << std::endl;
    }
}

template<typename TypeGold, typename TypeTest>
void Mat_CompareLt(TypeGold &gold, TypeTest &test)
{
    BASIC_ASSERT(gold.rows() == test.rows());
    BASIC_ASSERT(gold.cols() == test.cols());

    for (long r = 0; r < gold.rows(); r++) {
        for (long c = 0; c < gold.cols(); c++) {
            if (test(r, c) != gold(r, c)) {
                printf("[FAILED] Compare Failed: (r:%ld, c:%ld)[rols:%ld, cols:%ld]\n",
                    r, c, gold.rows(), gold.cols());
                std::cout << "Gold is: " << gold(r, c) << std::endl;
                std::cout << "Test is: " << test(r, c) << std::endl;
                std::cout << "Diff is: " << gold(r, c) - test(r, c) << std::endl;
                return;
            }
        }
    }
}

template<typename TypeFrom>
void Mat_RandFloat0to1(TypeFrom &from)
{
    for (long r = 0; r < from.rows(); r++) {
        for (long c = 0; c < from.cols(); c++) {
            from(r, c) = RandFloat0to1<typename std::remove_reference<decltype(from(0, 0))>::type>();
            //from(r, c) = RandFloat0to1<typename TypeFrom::Scalar_t>();
        }
    }
}