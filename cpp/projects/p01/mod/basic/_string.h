#pragma once

#include <string>
#include "_basic.h"

template<typename T>
void StringAppendInt(std::string &str, T num, T indents = 1)
{
    BASIC_ASSERT(indents < 100);
    char *buf = (char *)calloc(100, 1);

    T remain;
    T ctr = 0;
    T i;

    for (i = 98; i >= 0; i--) {
        remain = num % 10;
        num = num / 10;

        buf[i] = '0' + remain;
        ctr++;

        if (ctr >= indents) {
            if (num == 0) {
                break;
            }
        }
    }

    BASIC_ASSERT(i >= 0);
    BASIC_ASSERT(i <= 98);
    str.append((const char *)&buf[i]);

    free(buf);
}