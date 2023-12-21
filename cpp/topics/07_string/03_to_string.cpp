#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstring>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <unordered_map>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s() %d\n", __func__, __LINE__);

//
//
//

static int ctr = 0;

int main()
{
    std::string myString = "abc" + std::to_string(ctr);

    COUT(myString);
    return 0;
}