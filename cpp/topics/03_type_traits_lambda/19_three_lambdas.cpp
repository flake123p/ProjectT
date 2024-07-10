
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
00000000000011be l     F .text	0000000000000013              main::{lambda()#2}::operator()() const
00000000000011d2 l     F .text	0000000000000013              main::{lambda()#3}::operator()() const

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
    11bd:	90                   	nop

00000000000011be <main::{lambda()#2}::operator()() const>:
main::{lambda()#2}::operator()() const:
    11be:	55                   	push   %rbp
    11bf:	48 89 e5             	mov    %rsp,%rbp
    11c2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    11c6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11ca:	8b 00                	mov    (%rax),%eax
    11cc:	83 c0 02             	add    $0x2,%eax
    11cf:	5d                   	pop    %rbp
    11d0:	c3                   	ret    
    11d1:	90                   	nop

00000000000011d2 <main::{lambda()#3}::operator()() const>:
main::{lambda()#3}::operator()() const:
    11d2:	55                   	push   %rbp
    11d3:	48 89 e5             	mov    %rsp,%rbp
    11d6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    11da:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11de:	8b 00                	mov    (%rax),%eax
    11e0:	83 c0 03             	add    $0x3,%eax
    11e3:	5d                   	pop    %rbp
    11e4:	c3                   	ret    

00000000000011e5 <main>:
main():
    11e5:	f3 0f 1e fa          	endbr64 
    11e9:	55                   	push   %rbp
    11ea:	48 89 e5             	mov    %rsp,%rbp
    11ed:	48 83 ec 20          	sub    $0x20,%rsp
    11f1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    11f8:	00 00 
    11fa:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    11fe:	31 c0                	xor    %eax,%eax
    1200:	c7 45 f4 01 00 00 00 	movl   $0x1,-0xc(%rbp)
    1207:	8b 45 f4             	mov    -0xc(%rbp),%eax
    120a:	89 45 e8             	mov    %eax,-0x18(%rbp)
    120d:	8b 45 f4             	mov    -0xc(%rbp),%eax
    1210:	89 45 ec             	mov    %eax,-0x14(%rbp)
    1213:	8b 45 f4             	mov    -0xc(%rbp),%eax
    1216:	89 45 f0             	mov    %eax,-0x10(%rbp)
    1219:	48 8d 45 e8          	lea    -0x18(%rbp),%rax
    121d:	48 89 c7             	mov    %rax,%rdi
    1220:	e8 85 ff ff ff       	call   11aa <main::{lambda()#1}::operator()() const>
    1225:	89 c6                	mov    %eax,%esi
    1227:	48 8d 05 de 0d 00 00 	lea    0xdde(%rip),%rax        # 200c <__pstl::execution::v1::unseq+0x1>
    122e:	48 89 c7             	mov    %rax,%rdi
    1231:	b8 00 00 00 00       	mov    $0x0,%eax
    1236:	e8 45 fe ff ff       	call   1080 <printf@plt>
    123b:	48 8d 45 ec          	lea    -0x14(%rbp),%rax
    123f:	48 89 c7             	mov    %rax,%rdi
    1242:	e8 77 ff ff ff       	call   11be <main::{lambda()#2}::operator()() const>
    1247:	89 c6                	mov    %eax,%esi
    1249:	48 8d 05 cb 0d 00 00 	lea    0xdcb(%rip),%rax        # 201b <__pstl::execution::v1::unseq+0x10>
    1250:	48 89 c7             	mov    %rax,%rdi
    1253:	b8 00 00 00 00       	mov    $0x0,%eax
    1258:	e8 23 fe ff ff       	call   1080 <printf@plt>
    125d:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
    1261:	48 89 c7             	mov    %rax,%rdi
    1264:	e8 69 ff ff ff       	call   11d2 <main::{lambda()#3}::operator()() const>
*/
int main()
{
    int x = 1;

    auto lam123 = [=]() -> int { return x+1; };
    auto lam124 = [=]() -> int { return x+2; };
    auto lam125 = [=]() -> int { return x+3; };

    printf("lam123() = %d\n", lam123());
    printf("lam124() = %d\n", lam124());
    printf("lam125() = %d\n", lam125());

    return 0;
}
