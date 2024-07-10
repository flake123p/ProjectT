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
    char *(*pf[2])(char *p);
    
    pf[0] = fun1;
    pf[1] = &fun2;

    cout << pf[0]((char *)"fun1") << endl;
    cout << pf[1]((char *)"fun2") << endl;

    return 0;
}