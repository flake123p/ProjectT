
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
00000000000011aa l     F .text	0000000000000021              
    main::{lambda()#1}::operator()() const

00000000000011aa <main::{lambda()#1}::operator()() const>:
main::{lambda()#1}::operator()() const:
    11aa:	55                   	push   %rbp
    11ab:	48 89 e5             	mov    %rsp,%rbp
    11ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    11b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11b6:	48 8b 00             	mov    (%rax),%rax
    11b9:	8b 10                	mov    (%rax),%edx
    11bb:	83 c2 01             	add    $0x1,%edx
    11be:	89 10                	mov    %edx,(%rax)
    11c0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11c4:	48 8b 00             	mov    (%rax),%rax
    11c7:	8b 00                	mov    (%rax),%eax
    11c9:	5d                   	pop    %rbp
    11ca:	c3                   	ret 

00000000000011cb <main>:
main():
    11cb:	f3 0f 1e fa          	endbr64 
    11cf:	55                   	push   %rbp
    11d0:	48 89 e5             	mov    %rsp,%rbp
    11d3:	48 83 ec 20          	sub    $0x20,%rsp
    11d7:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    11de:	00 00 
    11e0:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    11e4:	31 c0                	xor    %eax,%eax
    11e6:	c7 45 ec 01 00 00 00 	movl   $0x1,-0x14(%rbp)
    11ed:	48 8d 45 ec          	lea    -0x14(%rbp),%rax
    11f1:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    11f5:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    11f9:	48 89 c7             	mov    %rax,%rdi
    11fc:	e8 a9 ff ff ff       	call   11aa <main::{lambda()#1}::operator()() const>
*/
int main()
{
    int x = 1;

    auto lam123 = [&]() -> int { x++; return x; };

    printf("Ori X = %d, lam123() = %d\n", x, lam123());

    return 0;
}
