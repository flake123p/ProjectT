
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    C++11 introduces new smart pointers: 
      std::unique_ptr, 
      std::shared_ptr, 
      std::weak_ptr. 
      
      std::auto_ptr now becomes deprecated and then eventually removed in C++17.  !!!!!!!!!!!!!
    
    std::unique_ptr is a non-copyable, movable pointer that manages its own heap-allocated memory.

    Note: Prefer using the std::make_X helper functions as opposed to using constructors. 
    
    See the sections for std::make_unique(Cpp14) and std::make_shared(Cpp11).
*/
struct Foo{
  void bar() {printf("this = %p\n", this);};
};

void demo_unique()
{
  printf("%s():\n", __func__);
  std::unique_ptr<Foo> p1 { new Foo{} };  // `p1` owns `Foo`
  if (p1) {
    p1->bar();
  }

  {
    std::unique_ptr<Foo> p2 {std::move(p1)};  // Now `p2` owns `Foo`
    //std::unique_ptr<Foo> p3 = p1;                                           // !!!ERROR!!!

    //f(*p2);
    p2->bar();

    p1 = std::move(p2);  // Ownership returns to `p1` -- `p2` gets destroyed
  }

  if (p1) {
    p1->bar();
  }
  // `Foo` instance is destroyed when `p1` goes out of scope
}

/*
    A std::shared_ptr is a smart pointer that manages a resource that is shared across multiple owners. 
    
    A shared pointer holds a control block which has a few components such as the managed object and a reference counter.  !!!REF COUNTER!!!
    
    All control block access is thread-safe, however, manipulating the managed object itself is not thread-safe.
*/
struct T {
  void bar() {printf("this = %p\n", this);};
};
void fooX(std::shared_ptr<T> t) {
  // Do something with `t`...
  t->bar();
}

void barX(std::shared_ptr<T> t) {
  // Do something with `t`...
  t->bar();
}

void bazX(std::shared_ptr<T> t) {
  // Do something with `t`...
  t->bar();
}

void demo_shared()
{
  printf("%s():\n", __func__);
  std::shared_ptr<T> p1 {new T{}};
  std::shared_ptr<T> p2 = p1;
  std::shared_ptr<T> p3 = p1;

  // Perhaps these take place in another threads?
  fooX(p1);
  barX(p2);
  bazX(p3);
}
int main()
{
  demo_unique();
  demo_shared();

  return 0;
}
