#include <cstdlib>
#include <iostream>
#include <cstring>

using namespace std;

char* fun1(char *p)
{
    return p;
}

char* fun2(char *p)
{
    return p;
}

int main() 
{
    char *(*a[2])(char *p);
    char *(*(*pf)[2])(char *p);

    pf = &a;
    a[0] = fun1;
    a[1] = &fun2;

    cout << (*pf)[0]((char *)"fun1") << endl;
    cout << pf[0][1]((char *)"fun2") << endl;

    return 0;
}