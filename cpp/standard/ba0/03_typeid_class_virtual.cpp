
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

using namespace std;

/*
    https://docs.microsoft.com/zh-tw/cpp/cpp/typeid-operator?view=msvc-160
*/

class Base {
public:
   virtual void vvfunc() {}; // exist virtual function
   virtual ~Base(){}; // for compile warning
};

class Derived : public Base {};

void class_name_w_virtual() {
   Derived* pd = new Derived;
   Base* pb = pd;
   cout << "should prints : [class Base *],    result =" << typeid( pb ).name() << endl;
   cout << "should prints : [class Derived],   result =" << typeid( *pb ).name() << endl; //!!!NOTICE!!!
   cout << "should prints : [class Derived *], result =" << typeid( pd ).name() << endl;
   cout << "should prints : [class Derived],   result =" << typeid( *pd ).name() << endl;
   delete pd;
}

class Base2 {};
class Derived2 : public Base2 {};
void class_name_w_o_virtual() {
   Derived2* pd = new Derived2;
   Base2* pb = pd;
   cout << "should prints : [class Base2 *],    result =" << typeid( pb ).name() << endl;
   cout << "should prints : [class Base2],      result =" << typeid( *pb ).name() << endl;
   cout << "should prints : [class Derived2 *], result =" << typeid( pd ).name() << endl;
   cout << "should prints : [class Derived2],   result =" << typeid( *pd ).name() << endl;
   delete pd;
}

int main() {
    class_name_w_virtual();
    class_name_w_o_virtual();
}