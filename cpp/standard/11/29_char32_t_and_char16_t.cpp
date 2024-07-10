
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Provides standard types for representing UTF-8 strings.
*/

char32_t utf8_strA[] = U"\u0123";
char16_t utf8_strB[] = u"\u0123";

int main()
{
  std::cout << utf8_strA << std::endl;
  std::cout << utf8_strB << std::endl;

  return 0;
}
