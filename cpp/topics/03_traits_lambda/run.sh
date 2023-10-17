#!/bin/bash

#g++ -std=c++17 -g $1 && ./a.out
g++ -std=c++17 $1 && ./a.out

objdump -SlzafphxgeGWtTrRs a.out > 1_ALL.log