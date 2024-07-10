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

/*
    To dump: objdump -Slz a.out > 0_Slz.log

    Conclusion: 
        1. constructor:        call _Z41__static_initialization_and_destruction_0ii()
        2. init() member func: call _Z41__static_initialization_and_destruction_0ii() too
        3. normal init() func: call _Z41__static_initialization_and_destruction_0ii() still ... !!!

*/
// struct A {
//     int num;
//     A() {
//         num = 99;
//     }
// };

// struct A {
//     int num;
//     void init() {
//         num = 99;
//     }
// };

struct A {
    int num;
};
void init(struct A &a) {
    a.num = 99;
}

int main()
{
    A a;
    init(a);
    return a.num;
}