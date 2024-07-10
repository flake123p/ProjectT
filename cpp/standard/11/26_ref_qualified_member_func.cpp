
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    Member functions can now be qualified depending on whether *this is an lvalue or rvalue reference.
*/

struct Bar {
  // ...
};

struct Foo {
  Bar getBar() & { printf("ver: &\n"); return bar; }
  Bar getBar() const& { printf("ver: const&\n"); return bar; }
  Bar getBar() && { printf("ver: &&\n");return std::move(bar); }
private:
  Bar bar;
};

int main()
{
  {
    Foo foo{};
    Bar bar = foo.getBar(); // calls `Bar getBar() &`

    const Foo foo2{};
    Bar bar2 = foo2.getBar(); // calls `Bar Foo::getBar() const&`

    Foo{}.getBar(); // calls `Bar Foo::getBar() &&`
    
    std::move(foo).getBar(); // calls `Bar Foo::getBar() &&`

    std::move(foo2).getBar(); // calls `Bar Foo::getBar() const&&`
  }

  return 0;
}
