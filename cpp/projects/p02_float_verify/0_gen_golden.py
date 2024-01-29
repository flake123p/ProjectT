import torch
import numpy as np
from py754 import *

#DEBUG_PRINT = True
DEBUG_PRINT = False

def log_print(hexStr, floatStr, idx, fout):
    if DEBUG_PRINT:
        print(hexStr, ',', floatStr, idx)
    else:
        the_str = '{ ' + hexStr + ', ' + floatStr + '},'
        print(the_str)
        fout.write(the_str + '\n')

def GenGolden_Template(dtype, init_var, loops, half_loops, w0, w1, fout, minE=-127):
    fout = open(fout, 'w')
    input = torch.tensor([init_var], dtype=dtype)
    for i in range(loops):
        if input[0] == torch.inf:
            continue
        if input[0] == -torch.inf:
            continue
        s, f, e = py754_elem_hex_str(input[0])
        fhex_str, hexStr, floatStr = torch_elem_to_dual_strings(input[0])
        # print(e, fhex_str, ',', hexStr, ',', floatStr)
        if e <= minE:
            continue
        log_print(hexStr, floatStr, i, fout)
        
        if i < half_loops - 1:
            input[0] = input[0] * w0
        else:
            input[0] = input[0] * w1

GOLDEN_FOLDER = 'golden_c/'

def GenF32_1_to_0():
    GenGolden_Template(dtype=torch.float32, init_var=1.1, loops=300, half_loops=150, w0=1.0/1.1, w1=1.0/2.1, fout=GOLDEN_FOLDER+'F32_1_to_0.txt')
def GenF32_N1_to_0():
    GenGolden_Template(dtype=torch.float32, init_var=-1.1, loops=300, half_loops=150, w0=1.0/1.1, w1=1.0/2.1, fout=GOLDEN_FOLDER+'F32_N1_to_0.txt')
def GenF32_1_to_INF():
    GenGolden_Template(dtype=torch.float32, init_var=1.1, loops=300, half_loops=150, w0=1.1, w1=1.7, fout=GOLDEN_FOLDER+'F32_1_to_INF.txt')
def GenF32_N1_to_NINF():
    GenGolden_Template(dtype=torch.float32, init_var=-1.1, loops=300, half_loops=150, w0=1.1, w1=1.7, fout=GOLDEN_FOLDER+'F32_N1_to_NINF.txt')

def GenF16_1_to_0():
    GenGolden_Template(dtype=torch.float16, init_var=1.1, loops=400, half_loops=200, w0=1.0/1.01, w1=1.0/1.05, fout=GOLDEN_FOLDER+'F16_1_to_0.txt', minE=-15)
def GenF16_N1_to_0():
    GenGolden_Template(dtype=torch.float16, init_var=-1.1, loops=400, half_loops=200, w0=1.0/1.01, w1=1.0/1.05, fout=GOLDEN_FOLDER+'F16_N1_to_0.txt', minE=-15)
def GenF16_1_to_INF():
    GenGolden_Template(dtype=torch.float16, init_var=1.1, loops=400, half_loops=200, w0=1.01, w1=1.05, fout=GOLDEN_FOLDER+'F16_1_to_INF.txt', minE=-15)
def GenF16_N1_to_NINF():
    GenGolden_Template(dtype=torch.float16, init_var=-1.1, loops=400, half_loops=200, w0=1.01, w1=1.05, fout=GOLDEN_FOLDER+'F16_N1_to_NINF.txt', minE=-15)

if __name__ == '__main__':
    GenF32_1_to_0()
    GenF32_1_to_INF()
    GenF32_N1_to_0()
    GenF32_N1_to_NINF()
    GenF16_1_to_0()
    GenF16_1_to_INF()
    GenF16_N1_to_0()
    GenF16_N1_to_NINF()
    # f = 0.0000000000000000000000000000000000000094226489688714717183662574046568028776902819166384980183747160
    # float_str = "{:.100f}".format(f)
    # print(f)
    # print(float_str)
    # print(float.hex(f))

    # f = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # float_str = "{:.100f}".format(f)
    # print(f)
    # print(float_str)
    # print(float.hex(f))

    # f = 1.100000023841857910156250000000000000000000000000000000000000
    # float_str = "{:.100f}".format(f)
    # print(f)
    # print(float_str)
    # print(float.hex(f))
    
