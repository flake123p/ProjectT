#pragma once

#include <stdlib.h>

template<typename T>
void free_safely(T &ptr) {
    if (ptr != nullptr) {
        free(ptr);
        ptr = nullptr;
    }
}