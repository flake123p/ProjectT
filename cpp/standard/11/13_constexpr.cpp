
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
void demo2();
void demo3();

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Constant expressions are expressions that are possibly evaluated by the compiler at compile-time.  !!!COMPILE TIME!!!

    Only non-complex computations can be carried out in a constant expression 
    (these rules are progressively relaxed in later versions). 
    
    Use the constexpr specifier to indicate the variable, function, etc. is a constant expression.
*/

constexpr int square(int x) {
    return x * x;
}

int square2(int x) {
    return x * x;
}

int main()
{
    int a = square(2);  // mov DWORD PTR [rbp-4], 4

    int b = square2(2); // mov edi, 2
                        // call square2(int)
                        // mov DWORD PTR [rbp-8], eax

/*
    In the previous snippet, notice that the computation when calling square is carried out at compile-time, 
    and then the result is embedded in the code generation, while square2 is called at run-time.
*/

    demo2();
    demo3();
    return 0;
}

void demo2()
{
    // constexpr values are those that the compiler can evaluate, but are not guaranteed to, at compile-time:

    const int x = 123;
    //constexpr const int& y = x; // error -- constexpr variable `y` must be initialized by a constant expression !!!ERROR!!!
}

// Constant expressions with classes:
struct Complex {
    constexpr Complex(double r, double i) : re{r}, im{i} { }
    constexpr double real() { return re; }
    constexpr double imag() { return im; }

private:
    double re;
    double im;
};

void demo3()
{
    constexpr Complex I(0, 1);
    printf("%f, %f\n", I.real(), I.imag());
}