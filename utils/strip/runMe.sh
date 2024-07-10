#!/bin/bash

#
# REF: https://ithelp.ithome.com.tw/articles/10196279
#
g++ -std=c++17 demo.cpp
strip a.out -o b.out
readelf -S b.out > b.log
readelf -S a.out > a.log

# $ unset GTK_PATH && meld a.log b.log  // Observe that symtab, strtab are removed

readelf -Ws b.out > b.Ws.log
readelf -Ws a.out > a.Ws.log