// Learn with Examples, 2020, MIT license

// Backup From: https://www.learnsystemc.com/basic/concurrency

/*
SystemC uses simulation processes to model concurrency. It's not true concurrent execution.
When multiple processes are simulated as running concurrently, only one is executed at a particular time. However, the simulated time remain unchanged until all concurrent processes finishes their current tasks.
Thus, these processes are running concurrently on the same "simulated time". This differs from e.g. the Go language, which is real concurrency.

Let's understand the simulated concurrency with a simple example.
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(CONCURRENCY) {
    SC_CTOR(CONCURRENCY) { // constructor
        SC_THREAD(thread1); // register thread1
        SC_THREAD(thread2); // register thread2
    }
    void thread1() {
        while(true) { // infinite loop
            std::cout << sc_time_stamp() << ": thread1" << std::endl;
            wait(2, SC_SEC); // trigger again after 2 "simulated" seconds
        }
    }
    void thread2() {
        while(true) {
            std::cout << "\t" << sc_time_stamp() << ": thread2" << std::endl;
            wait(3, SC_SEC);
        }
    }
};

int sc_main(int, char*[]) {
    CONCURRENCY concur("concur"); // define an object
    sc_start(10, SC_SEC); // run simulation for 10 seconds
    return 0;
}
