
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <unordered_set>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    These containers maintain average constant-time complexity for search, insert, and remove operations. 
    
    In order to achieve constant-time complexity, sacrifices order for speed by hashing elements into buckets. 
    
    There are four unordered containers:

        unordered_set
        unordered_multiset
        unordered_map
        unordered_multimap
*/

std::unordered_set<void *> myunordered_set;

int main()
{
    void *ptr;

    myunordered_set.insert(ptr = malloc(10));
    myunordered_set.insert(malloc(20));
    myunordered_set.insert(malloc(30));

    printf("ptr = %p\n", ptr);
    myunordered_set.erase(ptr);
    free(ptr);

    printf("malloc tracer:\n");
    for (const auto &s : myunordered_set) {
        //std::cout << s << " ";
        printf("%p\n", s);
        free(s);
    }

    printf("curr len = %ld\n", myunordered_set.size());
    myunordered_set.clear();
    printf("curr len = %ld\n", myunordered_set.size());

    return 0;
}
