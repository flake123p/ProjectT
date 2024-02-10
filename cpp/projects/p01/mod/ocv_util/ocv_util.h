#pragma once

#include "basic.h"

int OcvUtil_BytesToImag1(u8 *bytes, int width, int height, const char *fileName);
int OcvUtil_BytesToImag3(u8 *bytes, int width, int height, const char *fileName);

void ocv_util();