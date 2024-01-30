import torch
import numpy as np
from py754 import *


if __name__ == '__main__':
    input1 = torch.tensor([0.333984], dtype=torch.float16).cuda()
    input2 = torch.tensor([127.4375], dtype=torch.float16).cuda()
    output = input1 + input2
    print(input1)
    print(input2)
    print(output)
    exit()


    # input = torch.tensor([65460.0], dtype=torch.float32)
    # f16 = torch.tensor([1.0], dtype=torch.float16)
    
    # for i in range(80):
        
    #     input2 = input.type_as(f16)

    #     print(input2, ' ... ', input)
    #     input += 1
