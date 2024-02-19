#pragma once

#include "_lib.h"
#include <iomanip>    // std::setw and std::setfill
#include <type_traits>


template <typename T>
class Selection {
public:
    int *mask = nullptr;
    T *val = nullptr;
    int num = 0;
    T max_val;
    T min_val;
    int need_to_free_val;
    Selection(int input_num, T input_min_val, T input_max_val, T *val_from_outer = nullptr) {
        num = input_num;
        mask = (int *)calloc(num, sizeof(int));
        if (val_from_outer == nullptr) {
            val = (T *)calloc(num, sizeof(T));
            need_to_free_val = 1;
        } else {
            val = val_from_outer;
        }
        max_val = input_max_val;
        min_val = input_min_val;
    }
    ~Selection() {
        free_safely(mask);
        if (need_to_free_val) {
            free_safely(val);
        }
    }
    void init() {
        for (int i = 0; i < num; i++) {
            mask[i] = 1;
        }
    }
    int max(int start, int end) {
        T curr_max = min_val;
        int idx = -1;
        for (int i = start; i < end; i++) {
            if (mask[i] == 0) {
                continue;
            }
            // printf("%d, %f, %f\n", i, val[i], curr_max);
            if (val[i] > curr_max) {
                curr_max = val[i];
                idx = i;
            }
        }
        if (idx != -1) {
            mask[idx] = 0;
        }
        return idx;
    }
    int min(int start, int end) {
        T curr_min = max_val;
        int idx = -1;
        for (int i = start; i < end; i++) {
            if (mask[i] == 0) {
                continue;
            }
            if (val[i] < curr_min) {
                curr_min = val[i];
                idx = i;
            }
        }
        if (idx != -1) {
            mask[idx] = 0;
        }
        return idx;
    }
};