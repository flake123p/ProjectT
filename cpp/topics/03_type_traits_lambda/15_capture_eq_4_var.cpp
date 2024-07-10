
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
00000000000011aa l     F .text	000000000000002b              main::{lambda()#1}::operator()() const

00000000000011aa <main::{lambda()#1}::operator()() const>:
main::{lambda()#1}::operator()() const:
    11aa:	55                   	push   %rbp
    11ab:	48 89 e5             	mov    %rsp,%rbp
    11ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    11b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11b6:	8b 10                	mov    (%rax),%edx
    11b8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11bc:	8b 40 04             	mov    0x4(%rax),%eax
    11bf:	01 c2                	add    %eax,%edx
    11c1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11c5:	8b 40 08             	mov    0x8(%rax),%eax
    11c8:	01 c2                	add    %eax,%edx
    11ca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    11ce:	8b 40 0c             	mov    0xc(%rax),%eax
    11d1:	01 d0                	add    %edx,%eax
    11d3:	5d                   	pop    %rbp
    11d4:	c3                   	ret 

00000000000011d5 <main>:
main():
    11d5:	f3 0f 1e fa          	endbr64 
    11d9:	55                   	push   %rbp
    11da:	48 89 e5             	mov    %rsp,%rbp
    11dd:	48 83 ec 30          	sub    $0x30,%rsp
    11e1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    11e8:	00 00 
    11ea:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    11ee:	31 c0                	xor    %eax,%eax
    11f0:	c7 45 d0 01 00 00 00 	movl   $0x1,-0x30(%rbp)
    11f7:	c7 45 d4 0a 00 00 00 	movl   $0xa,-0x2c(%rbp)
    11fe:	c7 45 d8 64 00 00 00 	movl   $0x64,-0x28(%rbp)
    1205:	c7 45 dc e8 03 00 00 	movl   $0x3e8,-0x24(%rbp)
    120c:	8b 45 d0             	mov    -0x30(%rbp),%eax
    120f:	89 45 e0             	mov    %eax,-0x20(%rbp)
    1212:	8b 45 d4             	mov    -0x2c(%rbp),%eax
    1215:	89 45 e4             	mov    %eax,-0x1c(%rbp)
    1218:	8b 45 d8             	mov    -0x28(%rbp),%eax
    121b:	89 45 e8             	mov    %eax,-0x18(%rbp)
    121e:	8b 45 dc             	mov    -0x24(%rbp),%eax
    1221:	89 45 ec             	mov    %eax,-0x14(%rbp)
    1224:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
    1228:	48 89 c7             	mov    %rax,%rdi
    122b:	e8 7a ff ff ff       	call   11aa <main::{lambda()#1}::operator()() const>
*/
int main()
{
    int x = 1;
    int y = 10;
    int z = 100;
    int a = 1000;

    auto lam123 = [=]() -> int { return x+y+z+a; };

    printf("lam123() = %d\n", lam123());

    return 0;
}
