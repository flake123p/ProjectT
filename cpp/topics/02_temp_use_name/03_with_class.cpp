//
// Template argument deduction
//      https://en.cppreference.com/w/cpp/language/template_argument_deduction
//
// https://stackoverflow.com/questions/10872730/can-a-template-function-be-called-with-missing-template-parameters-in-c
//


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
#define PRINT_FUNC printf("%s()\n", __func__);

class BinaryFunction {
public:
    BinaryFunction() = default;
    void run() {
        printf("Class name = (%s)\n", typeid(*this).name());
    }
};

template <typename T, bool Dump, class BinaryFunction>
void mytemp(T a1, BinaryFunction binop)
{
    if (Dump) {
        printf("Do Dump:\n");
        COUT(typeid(T).name());
        COUT(sizeof(T));
        COUT(a1);
        binop.run();
    }
}

int main()
{
    {
        class BinaryFunction bi;
        mytemp<int, true>(3, bi);
    }

    return 0;
}