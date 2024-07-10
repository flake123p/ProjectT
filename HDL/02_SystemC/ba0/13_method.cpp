// Learn with Examples, 2020, MIT license

// Backup From: https://www.learnsystemc.com/basic/method

/*
Method:
  1. may have static sensitivity.
  2. only a method process, may call the function next_trigger to create dynamic sensitivity.
  2. cannot be made runnable as a result of an immediate notification executed by the process itself, regardless of the static sensitivity or dynamic sensitivity of the method process instance.

next_trigger():
  1. is a member function of class sc_module, a member function of class sc_prim_channel, and a non-member function.
  2. can be called from:
    a) a member function of the module itself,
    b) from a member function of a channel, or
    c) from any function that is ultimately called from a method process.

NOTE:
  1) any local variables declared within the method process will be destroyed on return from the process. Data members of the module should be used to store persistent state associated with the method process.

Recall the difference between SC_METHOD and SC_THREAD
  1. SC_METHOD(func): does not have its own thread of execution, consumes no simulated time, cannot be suspended, and cannot call code that calls wait()
  2. SC_THREAD(func): has its own thread of execution, may consume simulated time, can be susupended, and can call code that calls wait()
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(PROCESS) {
    SC_CTOR(PROCESS) { // constructor
        SC_THREAD(thread); // register a thread process
        SC_METHOD(method); // register a method process
        SC_METHOD(method2);
    }
    void thread() {
        int idx = 0; // declare only once
        while (true) { // loop forever
            std::cout << "thread"<< idx++ << " @ " << sc_time_stamp() << std::endl;
            wait(1, SC_SEC); // re-trigger after 1 s
        }
    }
    void method() {
        // notice there's no while loop here
        int idx = 0; // re-declare every time method is triggered
        std::cout << "method: " << idx++ << " @ " << sc_time_stamp() << std::endl;
        next_trigger(1, SC_SEC);
    }
    void method2() {
        std::cout << "method2: ... " << " @ " << sc_time_stamp() << std::endl;
    }
};

int sc_main(int, char*[]) {
    PROCESS process("process");
    sc_start(4, SC_SEC);
    return 0;
}
