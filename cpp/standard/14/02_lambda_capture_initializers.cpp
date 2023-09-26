
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

using namespace std;

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

/*
    C++ 14
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md#return-type-deduction

    Lambda capture initializers
    This allows creating lambda captures initialized with arbitrary expressions. 

    The name given to the captured value does not need to be related to any variables in the enclosing 
    scopes and introduces a new name inside the lambda body. 

    The initializing expression is evaluated when the lambda is created (not when it is invoked).
*/

int factory(int i) { return i * 10; }

int capture_and_copy(void) // original x has no change
{
    PRINT_FUNC;

    int x = 9;

    auto generator = [x] () mutable {return x++;}; // ********** DEMO SENSATION ********** //
    
    auto a = generator(); // == 0
    COUT(x);
    auto b = generator(); // == 1
    COUT(x);

    COUT(a);
    COUT(b);

    return 0;
}

int capture_and_reference(void) // original x has been changed
{
    PRINT_FUNC;

    int x = 9;

    auto generator = [&x] () mutable {return x++;}; // ********** DEMO SENSATION ********** //
    
    auto a = generator(); // == 0
    COUT(x);
    auto b = generator(); // == 1
    COUT(x);

    COUT(a);
    COUT(b);

    return 0;
}

int capture_and_initialize(void) // original x has no change
{
    PRINT_FUNC;
    int x = 9;

    auto generator = [x = 0] () mutable {return x++;}; // this would not compile without 'mutable' as we are modifying x on each call // ********** DEMO SENSATION ********** //
    COUT(x);
    auto a = generator(); // == 0
    COUT(x);
    auto b = generator(); // == 1
    COUT(x);

    COUT(a);
    COUT(b);

    return 0;
}

/*
    Because it is now possible to move (or forward) values into a lambda that could previously be only 
    captured by copy or reference we can now capture move-only types in a lambda by value. 
    
    Note that in the below example the p in the capture-list of task2 on the left-hand-side 
    of = is a new variable private to the lambda body and does not refer to the original p.
*/
void move_unique()
{
    PRINT_FUNC;

    auto p = std::make_unique<int>(1);
    COUT(*p);
    //auto task1 = [=] { *p = 5; }; // ERROR: std::unique_ptr cannot be copied
    // vs.
    auto task2 = [p = std::move(p)] { *p = 5; }; // OK: p is move-constructed into the closure object
    // the original p is empty after task2 is created
}

/*
    Using this reference-captures can have different names than the referenced variable.
*/
void different_names()
{
    PRINT_FUNC;

    auto x = 1;
    auto f = [&r = x, x = x * 10] {
        ++r;
        return r + x;
    };
    COUT(x);
    COUT(f()); // sets x to 2 and returns 12
    COUT(x);
}

int main(int argc, char *argv[])
{
    capture_and_copy();
    capture_and_reference();
    capture_and_initialize();
    move_unique();
    different_names();
    
    return 0;
}