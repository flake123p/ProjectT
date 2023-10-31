#include <cstdlib>
#include <iostream>
#include <cstring>

using namespace std;

typedef int (*FUNC)(int, int);

int max_value(int a, int b)
{
    return a > b ? a : b;   
}

int min_value(int a, int b)
{
    return a < b ? a : b;
}

FUNC find_function(char *name)
{
    if(!strcmp(name, "max"))
    {
        return max_value;
    }
    else if(!strcmp(name, "min"))
    {
        return min_value;
    }

    cout << "err";
    return NULL;
}

int main() 
{
    int (*p)(int, int);

    p = find_function((char *)"max");
    cout << "max = " << p(3, 5) << endl;
    
    p = find_function((char *)"min");
    cout << "min = " << p(3, 5) << endl;

    return 0;
}