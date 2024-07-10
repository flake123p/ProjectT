#include "Vdesign_under_test.h"
#include <memory>
#include <iostream>
#include <systemc>

using namespace sc_core;
SC_MODULE(Testbench) {
    sc_clock clk;
    sc_signal<bool> rst;
    sc_signal<bool> valid;
    sc_signal<unsigned> data;
public:
    SC_HAS_PROCESS(Testbench);
    Testbench(
        const sc_module_name &name,
        Vdesign_under_test *dut
    ): sc_module(name) {
        dut->clk(clk);
        dut->rst(rst);
        dut->valid(valid);
        dut->data(data);
        SC_THREAD(Reset);
        SC_THREAD(Monitor);
    }

    void Reset() {
        rst.write(true);  // initial begin rst = 1;
        wait(5.0, SC_NS); // #5
        rst.write(false); // rst = 1
        wait(5.0, SC_NS); // #5
        rst.write(true);  // rst = 0 end
    }

    void Monitor() {
        wait(rst.negedge_event());     // @(negedge rst)
        wait(rst.posedge_event());     // @(posedge rst)
        while (true) {                 // forever begin
            wait(clk.posedge_event()); // @(posedge clk)
            if (valid.read())
                std::cout << char(data.read()) << std::endl;
        }                              // end
    }
};

int sc_main(int, char**)
{
    std::unique_ptr<Vdesign_under_test> dut(new Vdesign_under_test("dut"));
    std::unique_ptr<Testbench> testbench(new Testbench("testbench", dut.get()));
    sc_start(100.0, SC_NS);
    return 0;
}