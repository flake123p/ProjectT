// Learn with Examples, 2020, MIT license

// Backup from: https://www.learnsystemc.com/basic/trace

/*
A trace file:
  1. records a time-ordered sequence of value changes during simulation.
  2. uses VCD (Value change dump) file format.
  3. can only be created and opened by sc_create_vcd_trace_file.
  4. may be opened during elaboration or at any time during simulation.
  5. contains values that can only be traced by sc_trace.
  6. shall be opened before values can be traced to that file, and values shall not be traced to a given trace file if one or more delta cycles have elapsed since opening the file.
  7. shall be closed by sc_close_vcd_trace_file. A trace file shall not be closed before the final delta cycle of simulation.
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(MODULE) {
    sc_signal<int> sig;
    SC_CTOR(MODULE) {
        SC_THREAD(writer);
    }
    void writer() {
        sig.write(2);
        wait(1, SC_PS);
        sig.write(4);
        wait(1, SC_PS);
        sig.write(6);
        wait(1, SC_PS);
    }
};
int sc_main(int, char*[]) {
    MODULE module("module");

    sc_trace_file* file = sc_create_vcd_trace_file("trace");
    sc_trace(file, module.sig, "sigX");
    sc_start(5, SC_PS);
    sc_close_vcd_trace_file(file);
    return 0;
}
