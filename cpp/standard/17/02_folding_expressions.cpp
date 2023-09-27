
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

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  Folding expressions:

  A fold expression performs a fold of a template parameter pack over a binary operator.

  An expression of the form (... op e) or (e op ...), where op is a fold-operator 
  and e is an unexpanded parameter pack, are called unary folds.

  An expression of the form (e1 op ... op e2), where op are fold-operators, 
  is called a binary fold. Either e1 or e2 is an unexpanded parameter pack, but not both.

*/
template <typename... Args>
bool logicalAnd(Args... args) {
    // Binary folding.
    return (true && ... && args);
}

template <typename... Args>
auto sum(Args... args) {
    // Unary folding.
    return (... + args);
}

int main() {
    bool b = true;
    bool& b2 = b;
    auto logical_and = logicalAnd(b, b2, true); // == true
    CDUMP(logical_and);

    auto sum_result = sum(1, 2, 3.0);
    CDUMP(sum_result);
    
    return 0;
}
