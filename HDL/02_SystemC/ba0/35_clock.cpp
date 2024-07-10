// Learn with Examples, 2020, MIT license

// Backup from: https://www.learnsystemc.com/basic/clock

/*
sc_clock s a predefined primitive channel derived from the class sc_signal and intended to model the behavior of a digital clock signal. 
The value and events associated with the clock are accessed through the interface sc_signal_in_if<bool>.

Constructor:
sc_clock(
  constchar*name_, // unique module name
  double period_v_, // the time interval between two consecutive transitions from false to true, also equal to the time interval between two consecutive transitions from true to false. Greater than zero, default is 1 nanosecond.
  sc_time_unit period_tu_, // time unit, used for period
  double duty_cycle_, // the proportion of the period during which the clock has the value true. Between 0.0 and 1.0, exclusive. Default is 0.5.
  double start_time_v_, // the absolute time of the first transition of the value of the clock (false to true or true to false). Default is zero.
  sc_time_unit start_time_tu_,
  bool posedge_first_ = true ); // if true, the clock is initialized to false, and changes to true at the start time. Vice versa. Default is true.
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(CLOCK) {
    sc_port<sc_signal_in_if<bool>> clk; // a port to access clock
    SC_CTOR(CLOCK) {
        SC_THREAD(thread); // register a thread process
        sensitive << clk; // sensitive to clock
        dont_initialize();
    }
    void thread() {
        while (true) {
            std::cout << sc_time_stamp() << ", value = " << clk->read() << std::endl; // print current clock value
            wait(); // wait for next clock value change
        }
    }
};

int sc_main(int, char*[]) {
    sc_clock clk("clk", 10, SC_SEC, 0.2, 10, SC_SEC, false); // 10s period, 8s false, 2s true, start at 10s, start at false.
    CLOCK clock("clock"); // instantiate module
    clock.clk(clk); // bind port
    sc_start(31, SC_SEC); // run simulation for 31 s
    return 0;
}
