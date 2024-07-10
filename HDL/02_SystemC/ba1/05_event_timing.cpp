
#include <systemc>
using namespace sc_core;

SC_MODULE(EVENT) {
    sc_event e; // declare an event
    SC_CTOR(EVENT) {
        SC_THREAD(trigger); //register a trigger process
        SC_THREAD(catcher); // register a catcher process
    }
    void trigger() {
        e.notify(SC_ZERO_TIME);
        e.notify(1, SC_SEC);   // no use ...
        e.notify(2, SC_SEC);   // no use ...
        wait(5, SC_SEC);
        e.notify(SC_ZERO_TIME);
    }
    void catcher() {
        while (true) { // loop forever
            wait(e); // wait for event
            std::cout << "Event cateched at " << sc_time_stamp() << std::endl; // print to console
        }
    }
};

int sc_main(int, char*[]) {
    EVENT event("event"); // define object
    sc_start(8, SC_SEC); // run simulation for 8 seconds
    return 0;
}
