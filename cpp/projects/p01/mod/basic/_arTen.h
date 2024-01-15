#pragma once
#include <iostream>
#include <vector>
#include "_basic.h"

template <typename T>
class ArTen {
public:
    size_t dims_;
    union {
        int prod_;
        int size;
    };
    std::vector<int> shape_;
    std::vector<int> stride_;
    T *array_; //storage

    ArTen(const std::initializer_list<int>& shape) {
        //printf("list.size() = %ld\n", shape.size());

        array_ = nullptr;
        
        dims_ = shape.size();
        BASIC_ASSERT(dims_ != 0);

        shape_.clear();
        stride_.clear();

        for (auto& e : shape) {
            shape_.push_back(e);
            stride_.push_back(1);
        }

        int prod = 1;
        int lastDim = 1;
        for (size_t i = stride_.size(); i != 0; i--) {
            prod = prod * lastDim;
            stride_[i-1] = prod;
            lastDim = shape_[i-1];
        }

        prod = prod * lastDim;
        prod_ = prod;
        //printf("Final prod = %d\n", prod);

        //array_ = (T *)malloc(sizeof(T) * prod);
        array_ = (T *)calloc(sizeof(T) * prod, 1);

        BASIC_ASSERT(shape_.size() == dims_);
        BASIC_ASSERT(stride_.size() == dims_);
    };

    ~ArTen() {
        if (array_ != nullptr) {
            free(array_);
            array_ = nullptr;
        }
    };

    int indexing(const std::initializer_list<int>& indices) {
        BASIC_ASSERT(indices.size() == dims_);

        size_t i = 0;
        int loc = 0;
        for (auto& e : indices) {
            BASIC_ASSERT(e >= 0);
            BASIC_ASSERT(e < shape_[i]);
            loc += e * stride_[i];
            i++;
        }

        // for (size_t i = 0; i < dims_; i++) {
        //     printf("%d\n", indices[i]);
        // }
        BASIC_ASSERT(loc < prod_);
        BASIC_ASSERT(loc >= 0);
        //printf("loc = %d\n", loc);
        return loc;
    };

    T &operator()(int x, int y, int z) {
        BASIC_ASSERT(dims_ == 3);
        BASIC_ASSERT(x < shape_[0]);
        BASIC_ASSERT(y < shape_[1]);
        BASIC_ASSERT(z < shape_[2]);
        return ref({x, y, z});
    };

    T &operator()(int row, int col) {
        BASIC_ASSERT(dims_ == 2);
        BASIC_ASSERT(row < shape_[0]);
        BASIC_ASSERT(col < shape_[1]);
        return ref({row, col});
    };

    T &operator()(int idx) {
        BASIC_ASSERT(idx < prod_);
        return array_[idx];
    };

    T &ref(const std::initializer_list<int>& indices) {
        return array_[indexing(indices)];
    };

    T get(const std::initializer_list<int>& indices) {
        return array_[indexing(indices)];
    };

    int set(const std::initializer_list<int>& indices, T input) {
        int loc = indexing(indices);
        array_[loc] = input;
        return 0;
    };

    void dump() {
        printf("dims = %lu\n", dims_);
        printf("prod = %d\n", prod_);

        printf("shape = ");
        for (auto& e : shape_) {
            printf("%d ", e);
        }
        printf("\n");

        printf("stride = ");
        for (auto& e : stride_) {
            printf("%d ", e);
        }
        printf("\n");
    };

    template <typename TravFunc>
    void travers_array(TravFunc f) {
        for (int i = 0; i < prod_; i++) {
            f(i, &array_[i]);
        }
    }
    /*
        Assignment Example (Type Casting) :

        at.travers_array([](int idx, float *inst) {
            *inst = (float)2266;
        });
    */
};