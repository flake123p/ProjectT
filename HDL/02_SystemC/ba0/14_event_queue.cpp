// Learn with Examples, 2020, MIT license

// Backup From: https://www.learnsystemc.com/basic/event_queue

/*
An event queue:
  1. has a member function notify(), same as an event
  2. is a hierarchical channel and can have multiple notifications pending, this differs from an event which can have only one outstanding notificaiton scheduled.
  3. can only be constructed during elaboration.
  4. does not support immediate notification

Member functions:
  1. void notify(double, sc_time_unit) or void notify(const sc_time&):
    a) zero time i.e. SC_ZERO_TIME: a delta notification
    b) non-zero time: notification scheduled relative to the simulation time when function notify is called.
  2. void cancel_all(): immediately delete every pending notification for this event queue object including both delta and timed notifications.
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(QUEUE) {
    sc_event e;
    sc_event_queue eq;
    SC_CTOR(QUEUE) {
        SC_THREAD(trigger);
        SC_THREAD(catch_e);
        sensitive << e; // catch_e() will be triggered by event e
        dont_initialize(); // don't run cach_e() during initialization phase
        SC_THREAD(catch_eq);
        sensitive << eq; // cach_eq() will be triggered by event queue eq
        dont_initialize(); // don't run catch_eq() during initialization phase
    }
    void trigger() {
        while (true) {
            e.notify(2, SC_SEC); // trigger e afer 2 s
            e.notify(1, SC_SEC); // trigger e after 1 s, replaces previous trigger
            eq.notify(2, SC_SEC); // trigger eq after 2 s
            eq.notify(2, SC_SEC); // trigger eq after 2 s, Flake: still available
            eq.notify(1, SC_SEC); // trigger eq after 1 s, both triggers available
            wait(10, SC_SEC); // another round
        }
    }
    void catch_e() {
        while (true) {
            std::cout << sc_time_stamp() << ": catches e" << std::endl;
            wait(); // no parameter --> wait for static sensitivity, i.e. e
        }
    }
    void catch_eq() {
        while (true) {
            std::cout << sc_time_stamp() << ": catches eq" << std::endl;
            wait(); // wait for eq
        }
    }
};

int sc_main(int, char*[]) {
    QUEUE queue("queue"); // instantiate object 
    sc_start(20, SC_SEC); // run simulation for 20 s
    return 0;
}