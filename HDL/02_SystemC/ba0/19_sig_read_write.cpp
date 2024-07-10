// Learn with Examples, 2020, MIT license

// Backup from: https://www.learnsystemc.com/basic/signal_readwrite

/*
sc_signal:
  1. is a predefined primitive channel intended to model the behavior of a single piece of wire carrying a digital electronic signal.
  2. uses the evaluate-update scheme to ensure deterministic behavior in the case of simultaneous read and write actions. We maintain a current and new value.
  3. its write() method will submit an update request if the new value is different from the current value.
  4. implements the sc_signal_inout_if<T> interface.

Constructors:
  1. sc_signal(): calls the base class constructor from its initializer list as: sc_prim_channel(sc_gen_unique_name("signal"))
  2. sc_signal(const char* name_): calls the base class constructor from its initializer list as: sc_prim_channel(name_)

Member functions:
  1. T& read() or operator const T& (): return a reference to the current value of the signal but shall not modify the state of the signal.
  2. void write(const T&): modifies the value of the signal such that the signal appears to have the new value (as returned by member function read) in the next delta cycle but not before then.
  3. operator=: equivalent to write()
  4. sc_event& default_event(), sc_event& value_changed_event(): return a reference to the value-changed event.
  5. bool event(): return true if and only if the value of the signal changed in the update phase of the immediately preceding delta cycle and at the current simulation time.

As compared to fifo:
  1. sc_signal has only one slot for read/write
  2. sc_signal triggers an update request only if the new value is different from the current value
  3. read from sc_signal won't remove the value

Other than execution phase, a sc_signal:
  1. may be written during elaboration to initialize the value of the signal.
  2. may be written from function sc_main during elaboration or while simulation is paused, that is, before or after the call to function sc_start.
*/
#include <systemc>
using namespace sc_core;

SC_MODULE(SIGNAL) {
    sc_signal<int> s;
    SC_CTOR(SIGNAL) {
        SC_THREAD(readwrite);
    }
    void readwrite() {
        s.write(3);
        std::cout << "s = " << s << "; " << s.read() << std::endl;
        wait(SC_ZERO_TIME);
        std::cout << "after delta_cycle, s = " << s.read() << std::endl;
        
        s = 4;
        s = 5;
        int tmp = s;
        std::cout << "s = " << tmp << std::endl;
        wait(SC_ZERO_TIME);
        std::cout << "after delta_cycle, s = " << s.read() << std::endl;
    }
};

int sc_main(int, char*[]) {
    SIGNAL signal("signal");
    signal.s = -1;
    sc_start();
    return 0;
}