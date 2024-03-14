#!/bin/bash

#g++ -std=c++17 -Wall -Wextra -Werror _list.cpp $1 ; ./a.out
#g++ --version
#g++ -I../../projects/p01/mod/basic -std=c++2b -Wall -Wextra $1 ; ./a.out


#g++ $1  -mmmx -msse2 -mssse3 -msse4.1 -mavx -mavx2 -mavx512dq -march=native; ./a.out

#
# <immintrin.h>
#
# https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
#
g++ $1 -march=native; ./a.out
