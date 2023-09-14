
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    A lightweight array-like container of elements created using a "braced list" syntax. 
    
    For example, { 1, 2, 3 } creates a sequences of integers, that has type std::initializer_list<int>. 
    
    Useful as a replacement to passing a vector of objects to a function.
*/

int sum(const std::initializer_list<int>& list) {
    int total = 0;
    for (auto& e : list) {
        total += e;
    }

    return total;
}

int main()
{
    auto list = {1, 2, 3};

    printf("sum(list)      = %d\n", sum(list));      // 6
    printf("sum({1, 2, 3}) = %d\n", sum({1, 2, 3})); // 6
    printf("sum({})        = %d\n", sum({}));        // 0

    std::initializer_list<int> list2 = {3, 4, 5};
    printf("sum(list2)     = %d\n", sum(list2));     // 12
    return 0;
}
