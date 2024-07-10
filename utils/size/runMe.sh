#!/bin/bash

g++ -std=c++17 demo.cpp

size a.out    >size_0_.log
size -A a.out >size_0_A.log
size -B a.out >size_1_B.log
size --format=Berkeley a.out >size_2_Ber.log
size --format=SysV a.out     >size_3_SysV.log
size --format=GUN a.out      >size_4_Gnu.log
size --format=GUN -x a.out   >size_5_Gnu_x.log
size --format=GUN -t a.out   >size_6_Gnu_t.log

rm a.out
g++ -std=c++17 -g demo.cpp
size --format=GUN -t a.out   >size_7_Gnu_t_debug.log


rm a.out
g++ -std=c++17 demo.cpp
size --format=SysV a.out     >size_80_SysV.log
size --format=SysV -t a.out  >size_81_SysV_t.log # Nothin change
rm a.out
g++ -std=c++17 -g demo.cpp
size --format=SysV a.out     >size_82_SysV_debug.log # dump debug sections finally