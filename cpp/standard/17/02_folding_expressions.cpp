
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

int demo_ant() {
    bool b = true;
    bool& b2 = b;
    auto logical_and = logicalAnd(b, b2, true); // == true
    CDUMP(logical_and);

    {
        printf("\nsum(1, 2):\n");
        auto sum_result = sum(1, 2);
        CDUMP(sum_result);
    }
    {
        printf("\nsum(1, 2, 3): (type will be integer)\n");
        auto sum_result = sum(1, 2, 3);
        CDUMP(sum_result);
    }
    {
        printf("\nsum(1, 2, 3.0): (type will be double)\n");
        auto sum_result = sum(1, 2, 3.0);
        CDUMP(sum_result);
    }
    return 0;
}

/*
    <<< CPP Reference >>> Fold Expressions

        https://en.cppreference.com/w/cpp/language/fold
    
    Syntax
        ( pack op ... )	(1)	
        ( ... op pack )	(2)	
        ( pack op ... op init )	(3)	
        ( init op ... op pack )	(4)	

        1) Unary right fold.
        2) Unary left fold.
        3) Binary right fold.
        4) Binary left fold.
    
    op	-	any of the following 32 binary operators: 
    
        + - * / % ^ & | = < > << >> += -= *= /= %= ^= &= |= <<= >>= == != <= >= && || , .* ->*. 
    
    In a binary fold, both ops must be the same.
    
    1) Unary right fold (E op ...) becomes (E1 op (... op (EN-1 op EN)))
    2) Unary left fold (... op E) becomes (((E1 op E2) op ...) op EN)
    3) Binary right fold (E op ... op I) becomes (E1 op (... op (EN−1 op (EN op I))))
    4) Binary left fold (I op ... op E) becomes ((((I op E1) op E2) op ...) op EN)


    
*/
template<typename... Args>
void printer(Args&&... args)
{
    (std::cout << ... << args) << '\n';
}

template<typename... Args>
void printer2(Args&&... args) // ...... VARIANT
{
    (printf("%d ", args), ...);
    printf(" ... [1] %d\n", __LINE__); // line number is still correct ... 111
    (printf("%d ", args), ...);
    printf(" ... [2] %d\n", __LINE__);
}

template<typename T, typename... Args>
void push_back_vec(std::vector<T>& v, Args&&... args)
{
    static_assert((std::is_constructible_v<T, Args&&> && ...));
    (v.push_back(std::forward<Args>(args)), ...);
}

// template<class T, std::size_t... dummy_pack>
// constexpr T bswap_impl(T i, std::index_sequence<dummy_pack...>)
// {
//     T low_byte_mask = (unsigned char)-1;
//     T ret{};
//     ([&]
//     {
//         (void)dummy_pack;
//         ret <<= CHAR_BIT;
//         ret |= i & low_byte_mask;
//         i >>= CHAR_BIT;
//     }(), ...);
//     return ret;
// }

// constexpr auto bswap(std::unsigned_integral auto i)
// {
//     return bswap_impl(i, std::make_index_sequence<sizeof(i)>{});
// }

void demo_CppRef() 
{
    {
        printer(1, 2, 3, "_abc");
        printer2(1, 2, 3);
    }
    {
        std::vector<int> v;
        push_back_vec(v, 6, 2, 45, 12);
        push_back_vec(v, 1, 2, 9);
        for (int i : v)
            std::cout << i << ' ';
        std::cout << '\n';
    }
    {
        // static_assert(bswap<std::uint16_t>(0x1234u) == 0x3412u);
        // static_assert(bswap<std::uint64_t>(0x0123456789abcdefull) == 0xefcdab8967452301ULL);
    }
}

int main() 
{
    //demo_ant();
    demo_CppRef();
    return 0;
}