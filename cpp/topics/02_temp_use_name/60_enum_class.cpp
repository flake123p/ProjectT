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

enum NotEnumClass : int8_t {
// #define DEFINE_ST_ENUM_VAL_(_1, n) n,
//   AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
// #undef DEFINE_ENUM_ST_ENUM_VAL_
    Undefined,
    NumOptions
};

enum class ScalarType : int8_t {
// #define DEFINE_ST_ENUM_VAL_(_1, n) n,
//   AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
// #undef DEFINE_ENUM_ST_ENUM_VAL_
    Undefined,
    NumOptions
};

enum class ScalarType2 : int16_t {
};

int main()
{
    COUT(sizeof(NotEnumClass));
    COUT(sizeof(ScalarType));
    COUT(sizeof(ScalarType2));
    COUT((int)ScalarType::NumOptions);

    return 0;
}