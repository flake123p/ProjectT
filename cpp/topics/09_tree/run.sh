#!/bin/bash

#g++ -std=c++17 -Wall -Wextra -Werror _list.cpp $1 ; ./a.out
g++ -I../../projects/p01/mod/basic -std=c++17 -Wall -Wextra _list.cpp $1 ; ./a.out
