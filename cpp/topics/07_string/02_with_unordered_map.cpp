#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstring>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <unordered_map>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s() %d\n", __func__, __LINE__);

//
//
//

#define REDUCE_MAX1_F32  "_ZNreduce_kernel4321234aaabbb1_"
#define REDUCE_MAX2_F32  "_ZNreduce_kernel4321234aaabbb2_"
#define REDUCE_MAX3_F32  "_ZNreduce_kernel4321234aaabbb3_"
#define REDUCE_MAX4_F32  "_ZNreduce_kernel4321234aaabbb4_"
#define REDUCE_MAX5_F32a "_ZNreduce_kernel4321234aaabbb5_"
#define REDUCE_MAX5_F32b "_ZNreduce_kernel9999999aaabbb5_"
#define REDUCE_MAX       "_ZNreduce_kernel"
int f1(void **args) {
    PRINT_FUNC;
    return 0;
}
int f2(void **args) {
    PRINT_FUNC;
    return 0;
}
int f3(void **args) {
    PRINT_FUNC;
    return 0;
}
int f4(void **args) {
    PRINT_FUNC;
    return 0;
}
int f5(void **args) {
    PRINT_FUNC;
    return 0;
}
typedef int (*KernelFunc_t)(void **args);
std::unordered_map<std::string, KernelFunc_t> callbackDB = {
    {REDUCE_MAX1_F32,          f1},
    {REDUCE_MAX2_F32,          f2},
    {REDUCE_MAX3_F32,          f3},
    {REDUCE_MAX4_F32,          f4},
    {REDUCE_MAX5_F32a,         f5},
};

int StartAndEndString_IsMatch(std::string &tar, const char *strStart, const char *strEnd, int dontCareLen, KernelFunc_t callback)
{
    const int strStartLen = strlen(strStart);
    const int strEndLen = strlen(strEnd);
    const int totalChar = strStartLen + dontCareLen + strEndLen;
    const char *ctar = tar.c_str();

    if (totalChar != strlen(ctar)) {
        return 0;
    }

    if (strncmp (strStart, ctar, strStartLen) != 0) {
        return 0;
    }

    if (strncmp (strEnd, ctar + strStartLen + dontCareLen, strEndLen) != 0) {
        return 0;
    }

    callbackDB.emplace (tar, callback);

    return 1;
}

int run(std::string tar)
{
    // 2nd chance for matching through StartAndEndString_IsMatch()
    for (int i = 0; i < 2; i++) {
        printf("[%d]run(): %s\n", i, tar.c_str());
        std::unordered_map<std::string, KernelFunc_t>::const_iterator it = callbackDB.find(tar);
        if ( it == callbackDB.end() ) {
            printf("...... NOT IMPLEMENT ...... : %s\n", tar.c_str());
            if (StartAndEndString_IsMatch(tar, "_ZNreduce_kernel", "aaabbb5_", 7, f5)) {
                continue;
            }
        } else {
            // if (para_it != budaFuncParamDB.end()) {
            //     RTPP_PRINT("[PA] %u %u %u / %u %u %u / %u %u %u / %u %u %u / %d\n",
            //         para_it->second->tid[0], para_it->second->tid[1], para_it->second->tid[2],
            //         para_it->second->bid[0], para_it->second->bid[1], para_it->second->bid[2],
            //         para_it->second->bDIm[0], para_it->second->bDIm[1], para_it->second->bDIm[2],
            //         para_it->second->gDim[0], para_it->second->gDim[1], para_it->second->gDim[2],
            //         para_it->second->wSize);
            // }
            it->second(nullptr);
        }
        break;
    }

    return 0;
}

int main()
{
    run(REDUCE_MAX1_F32);
    run(REDUCE_MAX2_F32);
    run(REDUCE_MAX3_F32);
    run(REDUCE_MAX4_F32);
    run("abc");
    run(REDUCE_MAX5_F32b);
    run(REDUCE_MAX5_F32b);
    return 0;
}