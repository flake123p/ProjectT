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

int main()
{
    std::unordered_map<int, std::string> mymap;

    mymap.emplace (0x10, "func_a");
    mymap.emplace (0x20, "func_b");

    COUT(mymap.size());

    mymap.emplace (0x20, "func_c");  // emplace with same key, total size remains
    COUT(mymap.size());

    mymap.emplace (0x30, "func_d");  // emplace with same key, total size remains
    COUT(mymap.size());


    return 0;
}