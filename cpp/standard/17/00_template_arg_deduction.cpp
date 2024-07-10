
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

    Template argument deduction for class templates:
    
    Automatic template argument deduction much like how it's done for functions, but now including class constructors.
*/
template <typename T = double>
struct MyContainer {
    T val;
    MyContainer() : val{} {}
    MyContainer(T val) : val{val} {}
    void go() {
        COUT(sizeof(T));
        COUT(typeid(val).name());
    };
};

int main() {
    MyContainer c1 {1}; // OK MyContainer<int>
    MyContainer c2;     // OK MyContainer<float>

    c1.go();
    c2.go();
    return 0;
}
