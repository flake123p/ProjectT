#!/bin/bash

#
# REF: https://stackoverflow.com/questions/34732/how-do-i-list-the-symbols-in-a-so-file
#
g++ -std=c++17 demo1.cpp
nm -gDC a.out > log.demo1.log

g++ -c demo2_shared_lib.cpp -fPIC
g++ -shared -o demo2_shared_lib.so demo2_shared_lib.o
nm -gDC demo2_shared_lib.o > log.demo2_1.o.log #  no symbols
nm -gDC demo2_shared_lib.so > log.demo2_2.so.log

#
# Visibility: https://stackoverflow.com/questions/22244428/hiding-symbol-names-in-library
#
g++ -c -fvisibility=hidden demo3_hidden.cpp -fPIC
g++ -shared -o demo3_hidden.so demo3_hidden.o
nm -gDC demo3_hidden.o > log.demo3_1.o.log #  no symbols
nm -gDC demo3_hidden.so > log.demo3_2.so.log

#
# attr==default
#
g++ -c -fvisibility=hidden demo4_hidden_default.cpp -fPIC
g++ -shared -o demo4_hidden_default.so demo4_hidden_default.o
nm -gDC demo4_hidden_default.o > log.demo4_1.o.log #  no symbols
nm -gDC demo4_hidden_default.so > log.demo4_2.so.log

#
# objcopy: https://stackoverflow.com/questions/9909528/how-can-i-remove-a-symbol-from-a-shared-object  //Flake: Not working!!!
#
# -N (--strip-symbol)
#
objcopy -N libfunc2 demo4_hidden_default.o
g++ -shared -o demo4_hidden_default.so demo4_hidden_default.o
nm -gDC demo4_hidden_default.so > log.demo5_1.objcopy_NOT_WORKING.log
# strip .so
objcopy -N libfunc2 demo4_hidden_default.so
nm -gDC demo4_hidden_default.so > log.demo5_2.objcopy_NOT_WORKING.log

#
# NOT WORKING: https://stackoverflow.com/questions/35569242/removing-a-specific-symbol-from-an-so-file
#
strip -N libfunc2 demo4_hidden_default.o
g++ -shared -o demo4_hidden_default.so demo4_hidden_default.o
nm -gDC demo4_hidden_default.so > log.demo6_1.strip_NOT_WORKING.log
# strip .so
strip -N libfunc2 demo4_hidden_default.so
nm -gDC demo4_hidden_default.so > log.demo6_2.strip_NOT_WORKING.log