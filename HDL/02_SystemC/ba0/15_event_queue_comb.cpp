// Learn with Examples, 2020, MIT license

// Backup From: https://www.learnsystemc.com/basic/event_queue_combined

/*
    1. Multiple event queues can be "OR"-ed to form the static sensitivity of a process.
        cannot use "AND" operation in static sensitivity.
    2. Event queue cannot be used as input to wait(), thus cannot be used in dynamic sensitivity.
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(QUEUE_COMBINED) {
    sc_event_queue eq1, eq2;
    SC_CTOR(QUEUE_COMBINED) {
        SC_THREAD(trigger);
        SC_THREAD(catcher);
        sensitive << eq1 << eq2; // eq1 "or" eq2, cannot "and"
        dont_initialize();
    }
    void trigger() {
        eq1.notify(1, SC_SEC); // eq1 at 1 s
        eq1.notify(2, SC_SEC); // eq1 at 2 s
        eq2.notify(2, SC_SEC); // eq2 at 2 s
        eq2.notify(3, SC_SEC); // eq2 at 3 s
    }
    void catcher() {
        while (true) {
            std::cout << sc_time_stamp() << ": catches trigger" << std::endl;
            wait(); // cannot use event queue in dynamic sensitivity
        }
    }
};

int sc_main(int, char*[]) {
    QUEUE_COMBINED combined("combined");
    sc_start();
    return 0;
}