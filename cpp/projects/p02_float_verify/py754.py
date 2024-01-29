'''
    Use Case:

    def output_c_golden_gen_data_Float2UInt(self, pytorch_tensor, file_out):
    
        flattened_data = pytorch_tensor.flatten()

        type_str = str(pytorch_tensor.dtype)

        for i in flattened_data:
            print_str = py754_synthesize_hex_str(float.hex(i.item()), type_str = type_str)
            file_out.write("" + print_str + ",\n")
'''
def py754_synthesize_hex_str(hex_str, type_str):
    s, f, e = py754_parse_hex_str(hex_str)
    fracValue = int(f, 16)
    #
    # Synthesize sign bit, 
    # s is 0 for positive
    # s is 1 for negative
    #
    if type_str == 'torch.float32':
        bits = 32
        fracBits = 23
        signShift = 31
        expoShift = 8 - 1
    elif type_str == 'torch.float16':
        bits = 16
        fracBits = 10
        signShift = 15
        expoShift = 5 - 1
    elif type_str == 'torch.bfloat16':
        bits = 16
        fracBits = 7
        signShift = 15
        expoShift = 8 - 1
    else:
        print("[FAILED] Unknown type_str in py754_synthesize_hex_str():", type_str)
        exit()
    
    #
    # FP64 is 52
    #
    fracShift = 52 - fracBits
    
    signValue = s << signShift
    fracValue = fracValue >> fracShift
    expoCmpl = (1 << expoShift) - 1
    
    if e == 0 and f == "0":
        e = 0
        fracValue = 0
    else:
        e = (e + expoCmpl) << fracBits
    
    total = signValue + e + fracValue
    
    if bits == 32:
        totalStr = '0x{:08X}'.format(total)
    elif bits == 16:
        totalStr = '0x{:04X}'.format(total)
    else:
        print("[FAILED] Unknown bits in py754_synthesize_hex_str():", bits)
        exit()

    #print(fracValue, hex(fracValue), expoCmpl, e, hex(e), totalStr)
    return totalStr

def py754_parse_hex_str(hex_str):
        s = hex_str.split('x')
        if len(s) != 2:
            print("[FAILED] Unknown hex string:", s, hex_str)
            exit()
        #print(s[0], s[1])
        signStr = s[0]
        s = s[1].split('.')
        if len(s) != 2:
            print("[FAILED] Unknown hex string:", s)
            exit()
        s = s[1].split('p')
        if len(s) != 2:
            print("[FAILED] Unknown hex string:", s)
            exit()
        
        if signStr == '0':
            sign = 0
        else:
            sign = 1
        
        fracStr = s[0]
        expoStr = s[1]
        expo = int(expoStr)
        return sign, fracStr, expo
        # print(sign, s[0], s[1], x)
        # x = 128
        # print(str(hex(x)) + frac)

def py754_elem_hex_str(elem):
    return py754_parse_hex_str(float.hex(elem.item()))

def torch_elem_to_hex_string(elem):
    return py754_synthesize_hex_str(float.hex(elem.item()), type_str = str(elem.dtype))


def torch_elem_to_dual_strings(elem):
    fhex_str = float.hex(elem.item())
    hex_str = torch_elem_to_hex_string(elem)
    float_str = "{:.100f}".format(elem.item())
    return fhex_str, hex_str, float_str