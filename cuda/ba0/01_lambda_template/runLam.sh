#!/bin/bash

nvcc -g -std=c++17 --extended-lambda --compiler-bindir=/usr/bin/g++ $1 && ./a.out
#nvcc -g -std=c++17  $1 && ./a.out

objdump -SlzafphxgeGWtTrRs a.out > log.1_ALL.log
#objdump -Slz a.out > log.0_Slz.log