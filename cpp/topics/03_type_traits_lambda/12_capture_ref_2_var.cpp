
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
00000000000011aa l     F .text	000000000000003c
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
    11c4:	48 8b 40 08          	mov    0x8(%rax),%rax
    11c8:	8b 10                	mov    (%rax),%edx
    11ca:	83 c2 01             	add    $0x1,%edx
    11cd:	89 10                	mov    %edx,(%rax)
    11cf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11d3:	48 8b 00             	mov    (%rax),%rax
    11d6:	8b 10                	mov    (%rax),%edx
    11d8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11dc:	48 8b 40 08          	mov    0x8(%rax),%rax
    11e0:	8b 00                	mov    (%rax),%eax
    11e2:	01 d0                	add    %edx,%eax
    11e4:	5d                   	pop    %rbp
    11e5:	c3                   	ret

00000000000011e6 <main>:
main():
    11e6:	f3 0f 1e fa          	endbr64 
    11ea:	55                   	push   %rbp
    11eb:	48 89 e5             	mov    %rsp,%rbp
    11ee:	48 83 ec 30          	sub    $0x30,%rsp
    11f2:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    11f9:	00 00 
    11fb:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    11ff:	31 c0                	xor    %eax,%eax
    1201:	c7 45 d8 01 00 00 00 	movl   $0x1,-0x28(%rbp)
    1208:	c7 45 dc e8 03 00 00 	movl   $0x3e8,-0x24(%rbp)
    120f:	48 8d 45 d8          	lea    -0x28(%rbp),%rax
    1213:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    1217:	48 8d 45 dc          	lea    -0x24(%rbp),%rax
    121b:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    121f:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
    1223:	48 89 c7             	mov    %rax,%rdi
    1226:	e8 7f ff ff ff       	call   11aa <main::{lambda()#1}::operator()() const>
*/
int main()
{
    int x = 1;
    int y = 1000;

    auto lam123 = [&]() -> int { x++; y++; return x+y; };

    printf("Ori X+Y = %d, lam123() = %d\n", x+y, lam123());

    return 0;
}
