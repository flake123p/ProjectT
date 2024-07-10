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
//
//
class A {
public:
    int val;
    int val2;
    
    // Copy assignment operator.
    int& operator[](int idx)
    {
        if (idx == 0)
            return val;
        return val2;
    }
};

int main()
{
    class A a;

    a[1] = 3;

    printf("a[1] = %d\n", a[1]);

    return 0;
}