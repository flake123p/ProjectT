
#include <iostream>
#include <cstdint>
#include <memory>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

    Template argument deduction for class templates
    Automatic template argument deduction much like how it's done for functions, but now including class constructors.
*/
template <typename T = double>
struct MyContainer {
    T val;
    MyContainer() : val{} {}
    MyContainer(T val) : val{val} {}
    void go() {
        std::cout<<"sizeof T = "<<sizeof(T)<<std::endl;
        std::cout<<"typeid T = "<<typeid(val).name()<<std::endl;
    };
};

void abxx()
{
    printf("abxx\n");
}

int main() {
    MyContainer c1 {1}; // OK MyContainer<int>
    MyContainer c2; // OK MyContainer<float>

    c1.go();
    c2.go();
    abxx();
    return 0;
}