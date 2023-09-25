
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    User-defined literals allow you to extend the language and add your own syntax.

    To create a literal, define a T operator "" X(...) { ... } function that returns a type T, with a name X.

    Note that the name of this function defines the name of the literal.

    Any literal names not starting with an underscore are reserved and won't be invoked.

    There are rules on what parameters a user-defined literal function should accept, according to what type the literal is called on.
*/

// `unsigned long long` parameter required for integer literal.
long long operator "" _celsius(unsigned long long tempCelsius) {
    return std::llround(tempCelsius * 1.8 + 32);
}

// `const char*` and `std::size_t` required as parameters.
int operator "" _int(const char* str, std::size_t) {
  return std::stoi(str);
}

int main()
{
    printf("24_celsius = %lld\n", 24_celsius);

    printf("123_int = %d\n", "123"_int); // == 123, with type `int`

    return 0;
}
