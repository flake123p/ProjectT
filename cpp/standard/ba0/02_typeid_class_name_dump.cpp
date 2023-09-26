
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
#include <typeinfo>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://en.cppreference.com/w/cpp/language/typeid
*/

/*
Usage:
    #include <typeinfo>

    typeid(your_class).name()
    or
    typeid(*this).name()
*/
class base {
public:
    base()
    {
        printf("Class name = (%s)\n", typeid(*this).name()); // weird
        std::cout << typeid(*this).name() << '\n';           // weird still
        
        {
            const char *className = typeid(*this).name(); // correct weird stuff
            className++;
            printf("className = (%s)\n\n", className);
        }
    };
};

class ClassName {
public:
    const char *name;
    ClassName(int skipFirstChar = 1)
    {
        name = typeid(*this).name();
        if (skipFirstChar) // correct weird stuff
            name++;
        printf("ClassName Constructor, typeid name = %s, pretty func = %s\n", name, __PRETTY_FUNCTION__);
    };
};

class MyClass : public ClassName {
public:
    MyClass(){printf("MyClass Constructor\n");};
};

int main(int argc, char *argv[])
{
    
    class base b;
    printf("Object b name = (%s)\n", typeid(b).name());
    
    class MyClass c;
    printf("MyClass c name = (%s ... parent name still ...)\n", c.name);
    
    return 0;
}