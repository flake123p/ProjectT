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
#include <unordered_map>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

//
// https://cplusplus.com/reference/string/string/find/
//
int main()
{
    std::string str("abc123funca");

    std::size_t found = str.find("123");

    COUT(found);

    found = str.find("1277");

    COUT(found);
    COUT(std::string::npos);

    if (found == std::string::npos) {
        printf("Not Found\n");
    }


    return 0;
}