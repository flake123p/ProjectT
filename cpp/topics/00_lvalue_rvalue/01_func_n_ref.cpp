
#include "all.hpp"

int func()
{
    return 3;
}

int g_value = 0;
int &func2()
{
    return g_value;
}

int main() 
{
    { // reference demo
        int x = 3;
        int &r = x;
        printf("(before) x = %d\n", x);
        r = 5;
        printf("(after)  x = %d\n", x);
    }
    {
        func(); //return rvalue
        printf("func() = %d\n", func());
    }
    {
        func2() = 7; //return lvalue        KEYPOINT
        printf("g_value = %d\n", g_value);
    }

    return 0;
}
