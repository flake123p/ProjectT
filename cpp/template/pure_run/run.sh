#!/bin/bash

g++ $1 && ./a.out

# ver2
#g++ *.cpp && ./a.out

# ver3, standards
#g++ -std=c++11 $1 && ./a.out