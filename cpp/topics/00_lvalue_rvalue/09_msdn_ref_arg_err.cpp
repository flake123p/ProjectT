
#include "all.hpp"

//
// https://learn.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2010/dd293668(v=vs.100)
//

#include <iostream>
#include <algorithm>

using namespace std;

struct W
{
   W(int&, int&) {}
};

struct X
{
   X(const int&, int&) {}
};

struct Y
{
   Y(int&, const int&) {}
};

struct Z
{
   Z(const int&, const int&) {}
};

template <typename T, typename A1, typename A2>
T* factory(A1& a1, A2& a2)
{
   return new T(a1, a2);
}

//
// For const ..., make factory() above ambiguous
//
template <typename T, typename A1, typename A2>
T* factory(A1 a1, A2 a2)
{
   return new T(a1, a2);
}

int main()
{
    {
        Z* pz = factory<Z>(2, 2);
    }

    {
        int a = 4, b = 5;
        W* pw = factory<W>(a, b);
    }

    return 0;
}
