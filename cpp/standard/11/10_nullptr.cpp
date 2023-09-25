
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    C++11 introduces a new null pointer type designed to replace C's NULL macro. 
    
    nullptr itself is of type std::nullptr_t and can be implicitly converted into pointer types, 
    and unlike NULL, not convertible to integral types except bool.
*/

void foo(int)
{

}

void foo(char*)
{

}

int main()
{
    //foo(NULL); // error -- ambiguous                     !!!NOTICE!!!
    foo(nullptr); // calls foo(char*)

    return 0;
}
