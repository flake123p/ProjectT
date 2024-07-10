
## Link
https://en.cppreference.com/w/cpp/language/operator_precedence

Markdown table special sign:
https://meta.stackoverflow.com/questions/335600/use-special-characters-like-in-markdown-table-dialects-in-documentation

Table Auto Format:
Linux: Ctrl + Shift + I
Win:   Alt + Shift + F

## C++ Operator Precedence

| Precedence | Operator                   | Description                                               | Associativity |
| ---------- | -------------------------- | --------------------------------------------------------- | ------------- |
| 1          | ::                         | Scope resolution                                          | L 2 R         |
| 2          | a++   a--                  | Suffix/postfix increment and decrement                    | L 2 R         |
| -          | type()   type{}            | Functional cast                                           | L 2 R         |
| -          | a()                        | Function call                                             | L 2 R         |
| -          | a[]                        | Subscript                                                 | L 2 R         |
| -          | .   ->                     | Member access                                             | L 2 R         |
| 3          | ++a   --a                  | Prefix increment and decrement                            | R 2 L         |
| -          | +a   -a                    | Unary plus and minus                                      | R 2 L         |
| -          | !   ~                      | Logical NOT and bitwise NOT                               | R 2 L         |
| -          | (type)                     | C-style cast                                              | R 2 L         |
| -          | *a                         | Indirection (dereference)                                 | R 2 L         |
| -          | &a                         | Address-of                                                | R 2 L         |
| -          | sizeof                     | Size-of[note 1]                                           | R 2 L         |
| -          | co_await                   | await-expression (C++20)                                  | R 2 L         |
| -          | new   new[]                | Dynamic memory allocation                                 | R 2 L         |
| -          | delete   delete[]          | Dynamic memory deallocation                               | R 2 L         |
| 4          | .*   ->*                   | Pointer-to-membe                                          | L 2 R         |
| 5          | a*b   a/b   a%b            | Multiplication, division, and remainder                   | L 2 R         |
| 6          | a+b   a-b                  | Addition and subtraction                                  | L 2 R         |
| 7          | <<   >>                    | Bitwise left shift and right shift                        | L 2 R         |
| 8          | <=>                        | Three-way comparison operator (since C++20)               | L 2 R         |
| 9          | <   <=   >   >=            | For relational operators < and ≤ and > and ≥ respectively | L 2 R         |
| 10         | ==   !=                    | For equality operators = and ≠ respectively               | L 2 R         |
| 11         | a&b                        | Bitwise AND                                               | L 2 R         |
| 12         | ^                          | Bitwise XOR (exclusive or)                                | L 2 R         |
| 13         | <code>\|</code>            | Bitwise OR (inclusive or)                                 | L 2 R         |
| 14         | &&                         | Logical AND                                               | L 2 R         |
| 15         | <code>\|\|</code>          | Logical OR                                                | L 2 R         |
| 16         | a?b:c                      | Ternary conditional[note 2]                               | R 2 L         |
| -          | throw                      | throw operator                                            | R 2 L         |
| -          | co_yield                   | yield-expression (C++20)                                  | R 2 L         |
| -          | =                          | Direct assignment (provided by default for C++ classes)   | R 2 L         |
| -          | +=   -=                    | Compound assignment by sum and difference                 | R 2 L         |
| -          | *=   /=   %=               | Compound assignment by product, quotient, and remainder   | R 2 L         |
| -          | <<=   >>=                  | Compound assignment by bitwise left shift and right shift | R 2 L         |
| -          | &=   ^=   <code>\|=</code> | Compound assignment by bitwise AND, XOR, and OR           | R 2 L         |
| 17         | ,                          | Comma                                                     | L 2 R         |