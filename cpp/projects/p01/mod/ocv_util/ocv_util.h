#pragma once

#include "basic.h"

int OcvUtil_BytesToImag1(byte *bytes, int width, int height, const char *fileName);
int OcvUtil_BytesToImag3(byte *bytes, int width, int height, const char *fileName);

void ocv_util();