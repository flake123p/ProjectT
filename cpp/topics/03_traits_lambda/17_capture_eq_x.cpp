
#include <iostream>
#include <ostream>
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
#include <variant>
#include <optional>
#include <any>
#include <filesystem>
#include <cstddef>
#include <set>
#include <random>
#include <iterator>
#include <charconv>
#include <cassert>

#include "FunctionTraits.h"

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    [] - captures nothing.
    [=] - capture local objects (local variables, parameters) in scope by value.
    [&] - capture local objects (local variables, parameters) in scope by reference.
    [this] - capture this by reference.
    [a, &b] - capture objects a by value, b by reference.

    By default, value-captures cannot be modified inside the lambda because the compiler-generated method is marked as const. 

    The mutable keyword allows modifying captured variables. 

    The keyword is placed after the parameter-list (which must be present even if it is empty).
*/
/*
    https://zh-blog.logan.tw/2020/02/17/cxx-17-lambda-expression-capture-dereferenced-this/

    [&]
    [=]
    [&, this]
    [=, *this]
    [=, this]
*/

/*
00000000000011aa l     F .text	0000000000000013              main::{lambda()#1}::operator()() const

00000000000011aa <main::{lambda()#1}::operator()() const>:
main::{lambda()#1}::operator()() const:
    11aa:	55                   	push   %rbp
    11ab:	48 89 e5             	mov    %rsp,%rbp
    11ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    11b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11b6:	8b 00                	mov    (%rax),%eax
    11b8:	83 c0 01             	add    $0x1,%eax
    11bb:	5d                   	pop    %rbp
    11bc:	c3                   	ret

00000000000011bd <main>:
main():
    11bd:	f3 0f 1e fa          	endbr64 
    11c1:	55                   	push   %rbp
    11c2:	48 89 e5             	mov    %rsp,%rbp
    11c5:	48 83 ec 10          	sub    $0x10,%rsp
    11c9:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    11d0:	00 00 
    11d2:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    11d6:	31 c0                	xor    %eax,%eax
    11d8:	c7 45 f4 01 00 00 00 	movl   $0x1,-0xc(%rbp)
    11df:	8b 45 f4             	mov    -0xc(%rbp),%eax
    11e2:	89 45 f0             	mov    %eax,-0x10(%rbp)
    11e5:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    11e9:	48 89 c7             	mov    %rax,%rdi
    11ec:	e8 b9 ff ff ff       	call   11aa <main::{lambda()#1}::operator()() const>
*/
int main()
{
    int x = 1;

    auto lam123 = [x]() -> int { return x+1; };

    printf("Ori X = %d, lam123() = %d\n", x, lam123());

    return 0;
}
