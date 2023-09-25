
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Attributes provide a universal syntax over __attribute__(...), __declspec, etc.
*/

// `noreturn` attribute indicates `f` doesn't return.
[[ noreturn ]] void f() {
    throw "error";
}

int main()
{
    f();

    return 0;
}
