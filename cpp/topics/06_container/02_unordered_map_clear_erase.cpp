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
// https://cplusplus.com/reference/unordered_map/unordered_map/erase/
//
int main()
{
    std::unordered_map<std::string,std::string> mymap;

    // populating container:
    mymap["U.S."] = "Washington";
    mymap["U.K."] = "London";
    mymap["France"] = "Paris";
    mymap["Russia"] = "Moscow";
    mymap["China"] = "Beijing";
    mymap["Germany"] = "Berlin";
    mymap["Japan"] = "Tokyo";

    COUT(mymap.size());

    // erase examples:
    mymap.erase ( mymap.begin() );      // erasing by iterator
    COUT(mymap.size());

    mymap.erase ("France");             // erasing by key
    COUT(mymap.size());

    // show content:
    for ( auto& x: mymap )
        std::cout << x.first << ": " << x.second << std::endl;
    printf("\n");
    /*
        Notice that unordered_map containers do not follow any particular order to organize its elements, 
        therefore the effect of range deletions may not be easily predictable.
    */
    mymap.erase ( mymap.find("U.K."), mymap.end() ); // erasing by range
    COUT(mymap.size());

    // show content:
    for ( auto& x: mymap )
        std::cout << x.first << ": " << x.second << std::endl;
    printf("\n");

    //
    // Clear
    //
    printf("Clear:\n");
    mymap.clear();
    COUT(mymap.size());
    return 0;
}