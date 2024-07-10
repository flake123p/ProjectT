
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

    Declaring non-type template parameters with auto:
    
    Following the deduction rules of auto, while respecting the non-type template parameter list of allowable types[*], 
    template arguments can be deduced from the types of its arguments:

*/
template <auto... seq>
struct my_integer_sequence {
  // Implementation here ...

  // Flake: How to use??
  constexpr static int value = sizeof...(seq);

  void run() {
    const auto s = {seq...};
    for (auto i : s) {
      printf("%d\n", i);
    }
  }
};

int main() {
    // Explicitly pass type `int` as template argument.
    auto seq = std::integer_sequence<int, 0, 1, 2>();
    // Type is deduced to be `int`.
    auto seq2 = my_integer_sequence<1, 3, 5>();
    auto seq3 = my_integer_sequence<1, 3, 6>();

    //* - For example, you cannot use a double as a template parameter type, which also makes this an invalid deduction using auto.
    printf("seq2.value = %d\n", seq2.value);
    printf("seq3.value = %d\n", seq3.value);
    
    seq2.run();
    seq3.run();

    //printf("Addresses: %p, %p\n", (void *)seq2.run, (void *)seq3.run);
    
    return 0;
}
