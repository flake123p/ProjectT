#!/bin/bash

nvcc -g -std=c++17 --extended-lambda --compiler-bindir=/usr/bin/g++ gpu.cu -o gpu.o -c
gcc -g -std=c++17 thrd.cpp -o thrd.o -c
gcc -g -std=c++17 main.cpp -o main.o -c
gcc -o a.out main.o thrd.o gpu.o -lcudart -lcuda -lcublas -lcublasLt -lstdc++
./a.out

# Device funtion in main, use nvcc in main.cpp
# nvcc -g -std=c++17 --extended-lambda --compiler-bindir=/usr/bin/g++ gpu.cu -o gpu.o -c
# nvcc -g -std=c++17 main.cu -o main.o -c
# nvcc -o a.out main.o gpu.o -lcudart -lcuda -lcublas -lcublasLt
# ./a.out
