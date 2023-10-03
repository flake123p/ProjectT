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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

enum TheEnum {
    a,
    b,
};

namespace ns {

enum TheEnum {
    a = 10,
    b,
};

class TheClass {
public:
    enum TheEnum {
        a = 20,
        b,
    };
};
}

//using TheEnum = ns::TheClass::TheEnum; // error: conflicting declaration
using _TheEnum = ns::TheClass::TheEnum;

int main()
{
    COUT(TheEnum::a);
    COUT(ns::TheEnum::a);
    COUT(ns::TheClass::TheEnum::a);
    COUT(_TheEnum::a);

    return 0;
}