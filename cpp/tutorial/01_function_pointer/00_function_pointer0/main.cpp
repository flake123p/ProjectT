#include <cstdlib>
#include <iostream>

using namespace std;

int max_value(int a, int b)
{
    return a > b ? a : b;   
}

int main()
{
    int (*func_ptr)(int, int);

    //func_ptr = max_value;
    func_ptr = &max_value;
    
    cout << func_ptr(3,6) << endl;
}