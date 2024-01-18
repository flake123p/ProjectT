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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

//
//
//
template<typename TypeF, typename TypeU, int ExpoBits, int FracBits>
struct Lite {
    union {
        TypeF f;
        TypeU u;
        struct {
            TypeU frac : FracBits;
            TypeU expo : ExpoBits;
            TypeU sign : 1;
        };
    };
};

template<typename TypeF, typename TypeU, int ExpoBits, int FracBits>
struct Derived : public Lite<TypeF, TypeU, ExpoBits, FracBits> {
    // union {
    //     TypeF f;
    //     TypeU u;
    //     struct {
    //         TypeU frac : FracBits;
    //         TypeU expo : ExpoBits;
    //         TypeU sign : 1;
    //     };
    // };
    const int expoCmpl = (1 << (ExpoBits-1)) - 1;

    void dump() {
        COUT(this->f);
        COUT(this->u);
    }
};

int main()
{
    Derived<float, uint32_t, 8, 23> f32, f32y;
    f32.f = 1.3;

    f32y.f = 0;
    f32y.u = f32.u;

    f32.dump();
    f32y.dump();

    return 0;
}