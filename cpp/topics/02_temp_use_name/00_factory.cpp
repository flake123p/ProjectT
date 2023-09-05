
#include "all.hpp"

#include <iostream>
#include <algorithm>

using namespace std;

struct X
{
    X(const int&, int&) {printf("X\n");}
};

struct Y
{
    Y(int&, const int&) {printf("Y\n");}
};

struct Z
{

};

template <typename T, typename A1, typename A2>
T* factory(A1& a1, A2& a2)
{
    printf("%ld, %ld\n", sizeof(A1), sizeof(A2));
    return new T();
}

int main()
{
    {
        int a = 4;
        long long int b = 5;
        Z* pw = factory<Z>(a, b);
        delete(pw);
    }

    return 0;
}
