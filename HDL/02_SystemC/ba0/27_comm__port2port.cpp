// Learn with Examples, 2020, MIT license

// Backup from: https://www.learnsystemc.com/basic/port2port

/*
So far we covered the cases of:
  1. connecting two processes of same module via channel:
      process1() --> channel --> process2() 
  2. connecting two processes of different modules via port and channel:
      module1::process1() --> module1::port1 --> channel --> module2::port2 --> module2::process2()
  3. connecting two processes of different modules via export:
      module1::process1() --> module1::channel --> module1::export1 --> module2::port2 --> module2::process2()

In all these cases, a channel is needed to connect the ports. There is a special case that allows a port to directly connect to a port of submodules. I.e.,
  module::port1 --> module::submodule::port2
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(SUBMODULE1) { // a submodule that writes to channel
    sc_port<sc_signal_out_if<int>> p;
    SC_CTOR(SUBMODULE1) {
        SC_THREAD(writer);
    }
    void writer() {
        int val = 1; // init value
        while (true) {
            p->write(val++); // write to channel through port
            wait(1, SC_SEC);
        }
    }
};
SC_MODULE(SUBMODULE2) { // a submodule that reads from channel
    sc_port<sc_signal_in_if<int>> p;
    SC_CTOR(SUBMODULE2) {
        SC_THREAD(reader);
        sensitive << p; // triggered by value change on the channel
        dont_initialize();
    }
    void reader() {
        while (true) {
            std::cout << sc_time_stamp() << ": reads from channel, val=" << p->read() << std::endl;
            wait(); // receives from channel through port
        }
    }
};
SC_MODULE(MODULE1) { // top-level module
    sc_port<sc_signal_out_if<int>> p; // port
    SUBMODULE1 sub1; // declares submodule
    SC_CTOR(MODULE1): sub1("sub1") { // instantiate submodule
        sub1.p(p); // bind submodule's port directly to parent's port
    }
};
SC_MODULE(MODULE2) {
    sc_port<sc_signal_in_if<int>> p;
    SUBMODULE2 sub2;
    SC_CTOR(MODULE2): sub2("sub2") {
        sub2.p(p); // bind submodule's port directly to parent's port
    }
};

int sc_main(int, char*[]) {
    MODULE1 module1("module1"); // instantiate module1
    MODULE2 module2("module2"); // instantiate module2
    sc_signal<int> s; // define channel outside module1 and module2
    module1.p(s); // bind module1's port to channel, for writing purpose
    module2.p(s); // bind module2's port to channel, for reading purpose
    sc_start(2, SC_SEC);
    return 0;
}
