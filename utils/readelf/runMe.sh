#!/bin/bash

g++ -std=c++17 demo.cpp

readelf -a  a.out>readelf_0_a.log              # all dump !!!!
readelf -h  a.out>readelf_1_h.log
readelf -l  a.out>readelf_2_l.log
readelf -Ws a.out>readelf_3_Ws.log             # Symbol dump
readelf -Ws a.out | c++filt >readelf_4_Ws.log  # Symbol dump & c++filt

g++ -std=c++17 -g demo.cpp

readelf -a  a.out>readelf_0g_a.log
readelf -h  a.out>readelf_1g_h.log
readelf -l  a.out>readelf_2g_l.log
readelf -Ws a.out>readelf_3g_Ws.log             # Symbol dump
readelf -Ws a.out | c++filt >readelf_4g_Ws.log  # Symbol dump & c++filt

#
# Compare With Debug Build:
#
# $ meld readelf_0*
# or
# $ unset GTK_PATH && meld readelf_0*


# REF: https://shengyu7697.github.io/linux-readelf/