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
    
    // Copy assignment operator.
    A& operator=(const A& other)
    {
        this->val = other.val;
        return *this;
    }
};

int main()
{
    class A a, b;
    a.val = 999;
    b.val = 777;
    COUT(a.val);
    COUT(b.val);
    a = b;
    COUT(a.val);

    return 0;
}