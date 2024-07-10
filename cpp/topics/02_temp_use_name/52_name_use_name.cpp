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
#include <typeinfo>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

//
// like static, better than static: Because it works on user define types ( https://blog.csdn.net/alexhu2010q/article/details/109100494 )
//
namespace at {
    namespace cuda{
        int ax = 4;
    }
}

namespace bt {
using namespace at::cuda;
}

int main()
{
    COUT(at::cuda::ax); // 1st way to access ax

    COUT(bt::ax); // 2nd way to access ax

    return 0;
}