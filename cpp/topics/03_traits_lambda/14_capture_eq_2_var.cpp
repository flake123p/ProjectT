
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
00000000000011aa l     F .text	0000000000000013
    main::{lambda()#1}::operator()() const

00000000000011aa <main::{lambda()#1}::operator()() const>:
main::{lambda()#1}::operator()() const:
    11aa:	55                   	push   %rbp
    11ab:	48 89 e5             	mov    %rsp,%rbp
    11ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    11b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11b6:	8b 00                	mov    (%rax),%eax
    11b8:	8d 50 01             	lea    0x1(%rax),%edx
    11bb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11bf:	8b 40 04             	mov    0x4(%rax),%eax
    11c2:	01 d0                	add    %edx,%eax
    11c4:	5d                   	pop    %rbp
    11c5:	c3                   	ret

00000000000011c6 <main>:
main():
    11c6:	f3 0f 1e fa          	endbr64 
    11ca:	55                   	push   %rbp
    11cb:	48 89 e5             	mov    %rsp,%rbp
    11ce:	48 83 ec 20          	sub    $0x20,%rsp
    11d2:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    11d9:	00 00 
    11db:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    11df:	31 c0                	xor    %eax,%eax
    11e1:	c7 45 e8 01 00 00 00 	movl   $0x1,-0x18(%rbp)
    11e8:	c7 45 ec e8 03 00 00 	movl   $0x3e8,-0x14(%rbp)
    11ef:	8b 45 e8             	mov    -0x18(%rbp),%eax
    11f2:	89 45 f0             	mov    %eax,-0x10(%rbp)
    11f5:	8b 45 ec             	mov    -0x14(%rbp),%eax
    11f8:	89 45 f4             	mov    %eax,-0xc(%rbp)
    11fb:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    11ff:	48 89 c7             	mov    %rax,%rdi
    1202:	e8 a3 ff ff ff       	call   11aa <main::{lambda()#1}::operator()() const>
*/
int main()
{
    int x = 1;
    int y = 1000;

    auto lam123 = [=]() -> int { return x+1+y; };

    printf("lam123() = %d\n", lam123());

    return 0;
}
