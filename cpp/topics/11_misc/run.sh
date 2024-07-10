#!/bin/bash

#g++ -std=c++17 -Wall -Wextra -Werror _list.cpp $1 ; ./a.out
#g++ --version
g++ -O2 -I../../projects/p01/mod/basic -std=c++2b -Wall -Wextra $1 ; ./a.out
