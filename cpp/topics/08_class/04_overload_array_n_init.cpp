#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <unordered_map>

#include "_basic.h"

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
// #define PRINT_FUNC printf("%s()\n", __func__);

//
//
//
template<typename T>
struct AryObject {
    T *buf;
    int num;
    AryObject(int input_num) {
        num = input_num;
        buf = new T[num];
    }
    ~AryObject() {
        delete[] buf;
    }

    T& operator[](int idx)
    {
        BASIC_ASSERT(idx < num);
        return buf[idx];
    }
};

template<typename T>
struct AryEmpty {
    T *buf = nullptr;
    int num;
    AryEmpty(int input_num) {
        num = input_num;
        buf = (T *)calloc(num, sizeof(T));
    }
    ~AryEmpty() {
        if (buf != nullptr) {
            free(buf);
            buf = nullptr;
        }
    }

    T& operator[](int idx)
    {
        BASIC_ASSERT(idx < num);
        return buf[idx];
    }
};


int num = 3;

// int x[num];
AryObject<int> x(num);
AryEmpty<int> y(num);

int main()
{
    x[2] = 99;
    printf("x[2] = %d\n", x[2]);
    printf("y[2] = %d\n", y[2]);
    return 0;
}