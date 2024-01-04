
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    The ... syntax creates a parameter pack or expands one. 
    
    A template parameter pack is a template parameter that accepts zero or more 
    template arguments (non-types, types, or templates). 
    
    A template with at least one parameter pack is called a variadic template.
*/

template <typename... T>
struct arity {
    constexpr static int value = sizeof...(T);
};

/*
    An interesting use for this is creating an initializer list from a 
    parameter pack in order to iterate over variadic function arguments.
*/
template <typename First, typename... Args>
auto sum(const First first, const Args... args) -> decltype(first) {
    const auto values = {first, args...};
    return std::accumulate(values.begin(), values.end(), First{0});
}

template <typename... Args>
auto another_sum(const Args... args) -> double {
    const auto values = {args...};
    return std::accumulate(values.begin(), values.end(), double{0});
}

void demo0()
{
    static_assert(arity<>::value == 0);
    static_assert(arity<char, short, int>::value == 3);

    printf("arity<>::value = %d\n", arity<>::value);
    printf("arity<char, short, int>::value = %d\n", arity<char, short, int>::value);

    printf("sum(1, 2, 3, 4, 5) = %d\n", sum(1, 2, 3, 4, 5)); // 15
    printf("sum(1, 2, 3)       = %d\n", sum(1, 2, 3));       // 6
    printf("sum(1.5, 2.0, 3.7) = %f\n", sum(1.5, 2.0, 3.7)); // 7.2
    printf("another_sum(1.5, 2.0, 3.7) = %f\n", another_sum(1.5, 2.0, 3.7)); // 7.2
}

//
// https://www.geeksforgeeks.org/variadic-function-templates-c/
//

// To handle base case of below recursive
// Variadic function Template
void traverse()
{
    std::cout << "I am empty function and "
            "I am called at last.\n";
}

template <typename T, typename... Types>
void traverse(T var1, Types... var2)
{
    const std::size_t n = sizeof...(Types); // https://stackoverflow.com/questions/12024304/c11-number-of-variadic-template-function-parameters

    std::cout << var1 << ", var2 size = " << n << std::endl;
 
    traverse(var2...);
}

void demo1_using_args_1_by_1()
{
    traverse(0.3);
    traverse(1.5, 2.0, 3.7);
}

int main()
{
    demo0();

    demo1_using_args_1_by_1();

    return 0;
}
