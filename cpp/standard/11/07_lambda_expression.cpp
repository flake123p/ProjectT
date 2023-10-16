
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    [] - captures nothing.
    [=] - capture local objects (local variables, parameters) in scope by value.
    [&] - capture local objects (local variables, parameters) in scope by reference.
    [this] - capture this by reference.
    [a, &b] - capture objects a by value, b by reference.

    By default, value-captures cannot be modified inside the lambda because the compiler-generated method is marked as const. 

    The mutable keyword allows modifying captured variables. 

    The keyword is placed after the parameter-list (which must be present even if it is empty).
*/
/*
    https://zh-blog.logan.tw/2020/02/17/cxx-17-lambda-expression-capture-dereferenced-this/

    [&]
    [=]
    [&, this]
    [=, *this]
    [=, this]
*/

int main()
{
    int x = 1;

    auto getX = [=] { return x; };
    getX(); // == 1
    printf("%d\n", getX());

    auto addX = [=](int y) { return x + y; };
    addX(1); // == 2
    printf("%d\n", addX(1));

    auto getXRef = [&]() -> int& { return x; };
    getXRef(); // int& to `x`
    printf("%d\n", getXRef());

    return 0;
}
