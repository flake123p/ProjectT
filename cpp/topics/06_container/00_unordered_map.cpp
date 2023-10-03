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

    std::cout << "mymap contains:" << std::endl;
    for (auto& x: mymap)
        std::cout << x.first << ": " << x.second << std::endl;

    std::cout << std::endl;

    // Search (or mapping)
    COUT(mymap[0x20]);

    // map of map
    std::unordered_map<std::string, int> mymap2;

    mymap2.emplace ("func_a", 0x00000003);
    mymap2.emplace ("func_b", 0x00000005);

    printf("\nString as key:\n");
    COUT(mymap2[std::string("func_b")]);

    printf("\nDouble Mapping:\n");
    COUT(mymap2[mymap[0x20]]);

    printf("\nSearching / Finding:\n");
    std::unordered_map<int, std::string>::const_iterator got = mymap.find(0x10);
    if ( got == mymap.end() )
        printf("not found\n");
    else
        printf("found : %d / %s\n", got->first, got->second.c_str());
    
    printf("\nSearching / Finding:\n");
    got = mymap.find(0x99);
    if ( got == mymap.end() )
        printf("not found\n");
    else
        printf("found : %d / %s\n", got->first, got->second.c_str());
    return 0;
}