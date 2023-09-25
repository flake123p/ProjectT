
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Constructors can now call other constructors in the same class using an initializer list.
*/

struct Foo {
  int foo;
  Foo(int foo) : foo{foo} {printf("1 argument: int\n");}
  Foo() : Foo(0) {printf("No argument\n");}
};

int main()
{
    printf("Init:\n");
    Foo foo;

    printf("foo.foo = %d\n", foo.foo);
    foo.foo; // == 0
    return 0;
}
