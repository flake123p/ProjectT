#!/bin/bash

nvcc -g -std=c++17 $1 && ./a.out

objdump -SlzafphxgeGWtTrRs a.out > log.1_ALL.log
#objdump -Slz a.out > log.0_Slz.log