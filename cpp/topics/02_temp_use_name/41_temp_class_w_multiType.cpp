//
// Template argument deduction
//      https://en.cppreference.com/w/cpp/language/template_argument_deduction
//
// https://stackoverflow.com/questions/10872730/can-a-template-function-be-called-with-missing-template-parameters-in-c
//


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
#include <climits>
#include <cfloat>
#include <typeinfo>
#include <cstring>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

typedef struct {
    uint8_t a[2];
} MY_F16_t;

void type_demo ()
{
    float a;
    int b;
    uint8_t c;
    uint16_t d;

    if (std::is_same<float, decltype(a)>::value) {
        printf(" a is float\n");
    }

    if (std::is_same<int, decltype(b)>::value) {
        printf(" b is int\n");
    }

    if (std::is_same<int32_t, decltype(b)>::value) {
        printf(" b is int32_t\n");
    }

    if (std::is_same<uint8_t, decltype(c)>::value) {
        printf(" c is uint8_t\n");
    }

    if (std::is_same<uint16_t, decltype(d)>::value) {
        printf(" d is uint16_t\n");
    }

    MY_F16_t myF16;
    if (std::is_same<MY_F16_t, decltype(myF16)>::value) {
        printf(" myF16 is MY_F16_t\n");
    }
}

typedef enum {
    CMD_DTYPE_NONE = 0,
    CMD_DTYPE_FP32,
    CMD_DTYPE_FP64,
    CMD_DTYPE_FP16,
    CMD_DTYPE_BP16,
    CMD_DTYPE_INT64,
    CMD_DTYPE_INT32,
    CMD_DTYPE_INT16,
    CMD_DTYPE_INT8,
    CMD_DTYPE_UINT64,
    CMD_DTYPE_UINT32,
    CMD_DTYPE_UINT16,
    CMD_DTYPE_UINT8,
} CMD_DATA_TYPE_t;

template<typename T>
int getDType(T a) {
    if (std::is_same<float, decltype(a)>::value) {
        return CMD_DTYPE_FP32;
    } else if (std::is_same<double, decltype(a)>::value) {
        return CMD_DTYPE_FP64;
    } else if (std::is_same<int8_t, decltype(a)>::value) {
        return CMD_DTYPE_INT8;
    } else if (std::is_same<int16_t, decltype(a)>::value) {
        return CMD_DTYPE_INT16;
    } else if (std::is_same<int32_t, decltype(a)>::value) {
        return CMD_DTYPE_INT32;
    } else if (std::is_same<int64_t, decltype(a)>::value) {
        return CMD_DTYPE_INT64;
    } else if (std::is_same<uint8_t, decltype(a)>::value) {
        return CMD_DTYPE_UINT8;
    } else if (std::is_same<uint16_t, decltype(a)>::value) {
        return CMD_DTYPE_UINT16;
    } else if (std::is_same<uint32_t, decltype(a)>::value) {
        return CMD_DTYPE_UINT32;
    } else if (std::is_same<uint64_t, decltype(a)>::value) {
        return CMD_DTYPE_UINT64;
    }
    return CMD_DTYPE_NONE;
}

template<typename T>
uint64_t toU64(T a) {
    uint64_t result = 0;
    int bytes = 0;
    if (std::is_same<float, decltype(a)>::value) {
        bytes = 4;
    } else if (std::is_same<double, decltype(a)>::value) {
        bytes = 8;
    } else if (std::is_same<int8_t, decltype(a)>::value) {
        bytes = 1;
    } else if (std::is_same<int16_t, decltype(a)>::value) {
        bytes = 2;
    } else if (std::is_same<int32_t, decltype(a)>::value) {
        bytes = 4;
    } else if (std::is_same<int64_t, decltype(a)>::value) {
        bytes = 8;
    } else if (std::is_same<uint8_t, decltype(a)>::value) {
        bytes = 1;
    } else if (std::is_same<uint16_t, decltype(a)>::value) {
        bytes = 2;
    } else if (std::is_same<uint32_t, decltype(a)>::value) {
        bytes = 4;
    } else if (std::is_same<uint64_t, decltype(a)>::value) {
        bytes = 8;
    }
    memcpy(&result, &a, bytes);
    return result;
}

template<typename MAIN_T, typename EXP_T>
int hostPow(MAIN_T *_dst, MAIN_T *_src, uint32_t batch_size, uint32_t stride, uint32_t element_count, EXP_T exp) 
{
    if (getDType(exp) == CMD_DTYPE_FP32) {
        printf(" exp is float\n");
    } else if (getDType(exp) == CMD_DTYPE_FP64) {
        printf(" exp is double\n");
    } else if (std::is_same<int8_t, decltype(exp)>::value) {
        printf(" exp is int8_t\n");
    } else if (std::is_same<int16_t, decltype(exp)>::value) {
        printf(" exp is int16_t\n");
    } else if (std::is_same<int32_t, decltype(exp)>::value) {
        printf(" exp is int32_t\n");
    } else if (std::is_same<int64_t, decltype(exp)>::value) {
        printf(" exp is int64_t\n");
    } else if (std::is_same<uint8_t, decltype(exp)>::value) {
        printf(" exp is uint8_t\n");
    } else if (std::is_same<uint16_t, decltype(exp)>::value) {
        printf(" exp is uint16_t\n");
    } else if (std::is_same<uint32_t, decltype(exp)>::value) {
        printf(" exp is uint32_t\n");
    } else if (std::is_same<uint64_t, decltype(exp)>::value) {
        printf(" exp is uint64_t\n");
    }

    return 0;
};


int main()
{
    type_demo();

    int x[3] = {1, 2, 7};
    int y[3];

    printf("0.5:\n");
    hostPow(x, y, 1, 0, 3, 0.5);

    printf("0.5f:\n");
    hostPow(x, y, 1, 0, 3, 0.5f);

    return 0;
}