#ifndef _VEC_HPP_INCLUDED_

#include <iostream>
#include <vector>

#define LEN(array) sizeof(array)/sizeof(array[0]) 

template <typename t>
auto VecMake(int shape, t *data) {
    std::vector<t> ret;
    for (int i = 0; i < shape; i++) {
        ret.push_back(*data);
        data++;
    }
    return std::move(ret);
};

template <typename t>
auto VecMake(int shape0, int shape1, t *data) {
    std::vector<std::vector<t>> ret0;
    for (int i = 0; i < shape0; i++) {
        std::vector<t> ret1;
        for (int j = 0; j < shape1; j++) {
            ret1.push_back(*data);
            data++;
        }
        ret0.push_back(std::move(ret1));
    }
    return std::move(ret0);
};

template <typename t>
void VecDump(std::vector<t> &m) {
    printf("%s():\n", __func__);
    for (int i = 0; i < m.size(); i++) {
        std::cout << m[i] << "\t";
    }
    printf("\n");
};

template <typename t>
void VecDump(std::vector<std::vector<t>> &m) {
    printf("%s():\n", __func__);
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            std::cout << m[i][j] << "\t";
        }
        printf("\n");
    }
};

#define _VEC_HPP_INCLUDED_
#endif//_VEC_HPP_INCLUDED_