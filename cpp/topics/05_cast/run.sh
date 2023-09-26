#!/bin/bash

g++ -I../include $1 && ./a.out



exit
=======================================================================================
MSVC:

https://learn.microsoft.com/en-us/cpp/cpp/casting-operators?view=msvc-170

dynamic_cast Used for conversion of polymorphic types.

static_cast Used for conversion of nonpolymorphic types.

const_cast Used to remove the const, volatile, and __unaligned attributes.

reinterpret_cast Used for simple reinterpretation of bits.

safe_cast Used in C++/CLI to produce verifiable MSIL.