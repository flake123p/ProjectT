
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Type-safe enums that solve a variety of problems with C-style enums including: 
    implicit conversions, inability to specify the underlying type, scope pollution.
*/

// Specifying underlying type as `unsigned int`
enum class Color : unsigned int { Red = 0xff0000, Green = 0xff00, Blue = 0xff };
// `Red`/`Green` in `Alert` don't conflict with `Color`
enum class Alert : bool { Red, Green };

int main()
{
    Color c = Color::Red;
    Alert aR = Alert::Red;
    Alert aG = Alert::Green;

    printf("c = 0x%x\n", (unsigned int)c);

    printf("aR = 0x%x\n", (unsigned int)aR);
    printf("aG = 0x%x\n", (unsigned int)aG);

    return 0;
}
