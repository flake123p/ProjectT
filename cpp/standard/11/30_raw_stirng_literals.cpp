
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    C++11 introduces a new way to declare string literals as "raw string literals". 
    
    Characters issued from an escape sequence (tabs, line feeds, single backslashes, etc.) 
    can be inputted raw while preserving formatting. 
    
    This is useful, for example, to write literary text, which might contain a lot of quotes or special formatting. 
    
    This can make your string literals easier to read and maintain.


    A raw string literal is declared using the following syntax:

        R"delimiter(raw_characters)delimiter"
    
    where:
      delimiter is an optional sequence of characters made of any source character except parentheses, backslashes and spaces.
      raw_characters is any raw character sequence; must not contain the closing sequence ")delimiter".
*/

// msg1 and msg2 are equivalent.
const char* msg1 = "\nHello,\n\tworld!\n";
const char* msg2 = R"(
Hello,
	world!
)";

int main()
{
  std::cout << msg1 << std::endl;
  std::cout << msg2 << std::endl;

  return 0;
}
