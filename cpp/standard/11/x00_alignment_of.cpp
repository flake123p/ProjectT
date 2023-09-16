#include <cstdint>
#include <iostream>
#include <type_traits>
void demo_can_vectorize_up_to();

//
// https://en.cppreference.com/w/cpp/types/alignment_of
//

struct A {};

struct B
{
    std::int16_t q;
};

struct C
{
    std::int8_t p;
    std::int8_t q;
};

struct D
{
    std::int8_t p;
    std::int16_t q;
};

struct E
{
    std::int16_t p;
    std::int8_t q;
};

struct F
{
    std::int16_t p;
    std::int16_t q;
};

int main()
{
    std::cout << std::alignment_of<A>::value << ' ';
    std::cout << std::alignment_of<B>::value << ' ';
    std::cout << std::alignment_of<C>::value << ' ';
    std::cout << std::alignment_of<D>::value << ' ';
    std::cout << std::alignment_of<E>::value << ' ';
    std::cout << std::alignment_of<F>::value << ' ';

    std::cout << std::endl;

    std::cout << std::alignment_of<int>() << ' '; // alt syntax
    std::cout << std::alignment_of<double>() << ' ';
    std::cout << std::alignment_of<double>::value << ' ';
    //std::cout << std::alignment_of_v<double> << '\n'; // c++17 alt syntax

    std::cout << std::endl;

    demo_can_vectorize_up_to();
}

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template<typename scalar_t>
inline int my_can_vectorize_up_to(char *pointer) {
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
    constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
    printf("address = %lu ... [%d/%d] ", address, vec4_alignment, vec2_alignment);
    if (address % vec4_alignment == 0) {
        return 4;
    } else if (address % vec2_alignment == 0) {
        return 2;
    }
    return 1;
}

#include <stdint.h>
void demo_can_vectorize_up_to()
{
    printf("--- demo_can_vectorize_up_to start ---\n");

    int8_t buf8[10];
    int16_t buf16[10];
    int32_t buf32[10];
    int64_t buf64[10];

    printf("int8_t . buf8  = %d\n", my_can_vectorize_up_to<int8_t>((char *)buf8));
    printf("int8_t . buf16 = %d\n", my_can_vectorize_up_to<int8_t>((char *)buf16));
    printf("int8_t . buf32 = %d\n", my_can_vectorize_up_to<int8_t>((char *)buf32));
    printf("int8_t . buf64 = %d\n", my_can_vectorize_up_to<int8_t>((char *)buf64));

    printf("int8_t . &buf8[0]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)buf8));
    printf("int8_t . &buf8[1]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[1]));
    printf("int8_t . &buf8[2]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[2]));
    printf("int8_t . &buf8[3]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[3]));
    printf("int8_t . &buf8[4]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[4]));
    printf("int8_t . &buf8[5]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[5]));
    printf("int8_t . &buf8[6]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[6]));
    printf("int8_t . &buf8[7]  = %d\n", my_can_vectorize_up_to<int8_t>((char *)&buf8[7]));

    printf("int16_t . buf8  = %d\n", my_can_vectorize_up_to<int16_t>((char *)buf8));
    printf("int16_t . buf16 = %d\n", my_can_vectorize_up_to<int16_t>((char *)buf16));
    printf("int16_t . buf32 = %d\n", my_can_vectorize_up_to<int16_t>((char *)buf32));
    printf("int16_t . buf64 = %d\n", my_can_vectorize_up_to<int16_t>((char *)buf64));

    printf("int32_t . buf8  = %d\n", my_can_vectorize_up_to<int32_t>((char *)buf8));
    printf("int32_t . buf16 = %d\n", my_can_vectorize_up_to<int32_t>((char *)buf16));
    printf("int32_t . buf32 = %d\n", my_can_vectorize_up_to<int32_t>((char *)buf32));
    printf("int32_t . buf64 = %d\n", my_can_vectorize_up_to<int32_t>((char *)buf64));

    aligned_vector<char, 16> x;
}