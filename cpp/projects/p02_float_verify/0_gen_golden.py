import torch
import numpy as np
from py754 import *


def GenF32_1_to_0():
    input = torch.tensor([1.1], dtype=torch.float32)
    for i in range(300):
        s, f, e = py754_elem_hex_str(input[0])
        fhex_str, hexStr, floatStr = torch_elem_to_dual_strings(input[0])
        # print(e, fhex_str, ',', hexStr, ',', floatStr)
        if e >= -127:
            print(hexStr, ',', floatStr)
        
        if i < 149:
            input[0] = input[0] / 1.1
        else:
            input[0] = input[0] / 2.1


if __name__ == '__main__':
    GenF32_1_to_0()