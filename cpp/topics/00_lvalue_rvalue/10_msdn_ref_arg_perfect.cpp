
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
T* factory(A1&& a1, A2&& a2)
{
   return new T(std::forward<A1>(a1), std::forward<A2>(a2));
   /*
      The purpose of the std::forward function is to forward the parameters of 
      the factory function to the constructor of the template class.
   */
}

int main()
{
    {
        Z* pz = factory<Z>(2, 2);
        delete pz;
    }

    {
        int a = 4, b = 5;
        W* pw = factory<W>(a, b);
        delete pw;
    }

    return 0;
}
