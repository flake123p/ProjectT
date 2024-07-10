#include "Vdesign_under_test.h"
#include <memory>
#include <iostream>

int main(int, char**)
{
    std::unique_ptr<Vdesign_under_test> dut(new Vdesign_under_test);
    dut->clk = 1; dut->rst = 1; dut->eval();
    dut->rst = 0; dut->eval();
    dut->rst = 1; dut->eval();
    for (int i = 0; i < 40; ++i) {
        dut->clk = 1 - dut->clk; dut->eval();
        if (dut->clk == 0 and dut->valid == 1)
            std::cout << char(dut->data) << std::endl;
    }
    return 0;
}