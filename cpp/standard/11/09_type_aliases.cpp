
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Semantically similar to using a typedef however, type aliases with using are easier to read and are compatible with templates.
*/

template <typename T>
using Vec = std::vector<T>; //             !!!NOTICE!!! compatible with templates.

int main()
{
    Vec<int> v; // std::vector<int>

    using String = std::string;
    String s {"foo"};

    return 0;
}
