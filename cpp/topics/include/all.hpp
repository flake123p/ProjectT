
#include <iostream>
#include <cstdint>
#include <cassert>
#include <memory>
#include <utility>
#include <stdint.h>
#include <algorithm>

#define ALL_VER 1002003

#ifndef PRLOC
#define PRLOC prloc(__FILE__, __LINE__);
static inline void prloc(const char *file, int line)
{
    printf("[PRLOC] file: %s, line %d\n", file, line);
}
#endif