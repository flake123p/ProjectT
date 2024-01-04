
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <climits>
#include <cfloat>
#include <cassert>

using namespace std;

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
    C++ 14
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md#return-type-deduction

    C++14 introduces the [[deprecated]] attribute to indicate that a unit (function, class, etc.) 
    is discouraged and likely yield compilation warnings. 
    
    If a reason is provided, it will be included in the warnings.
*/

[[deprecated]]
void old_method()
{

}

[[deprecated("Use new_method instead")]]
void legacy_method()
{

}

int main(int argc, char *argv[])
{
    old_method();
    legacy_method();

    return 0;
}
