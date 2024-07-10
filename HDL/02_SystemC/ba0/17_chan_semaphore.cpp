// Learn with Examples, 2020, MIT license

// Backup from: https://www.learnsystemc.com/basic/channel_semaphore

/*
A semaphore:
  1. is a predefined channel intended to model the behavior of a software semaphore used to provide limited concurrent access to a shared resource.
  2. has an integer value, the semaphore value, which is set to the permitted number of concurrent accesses when the semaphore is constructed.
    a) if the initial value is one, the semaphore is equivelent to a mutex.

Member functions:
  1. int wait():
    a) If the semaphore value is greater than 0, wait() shall decrement the semaphore value and return.
    b) If the semaphore value is equal to 0, wait() shall suspend until the semaphore value is incremented (by another process).
    c) Shall unconditionally return the value 0.
  2. int trywait():
    a) If the semaphore value is greater than 0, trywait() shall decrement the semaphore value and shall return the value 0.
    b) If the semaphore value is equal to 0, trywait() shall immediately return the value –1 without modifying the semaphore value.
  3. int post():
    a) shall increment the semaphore value.
    b) shall use immediate notification to signal the act of incrementing the semaphore value to any waiting processes.
    c) shall unconditionally return the value 0.
  4. int get_value(): shall return the semaphore value.
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(SEMAPHORE) {
    sc_semaphore s; // declares semaphore
    SC_CTOR(SEMAPHORE) : s(2) { // init semaphore with 2 resources
        SC_THREAD(thread_1); // register 3 threads competing for resources
        SC_THREAD(thread_2);
        SC_THREAD(thread_3);
    }
    void thread_1() {
        while (true) {
            if (s.trywait() == -1) { // try to obtain a resource
                s.wait(); // if not successful, wait till resource is available
            }
            std::cout<< sc_time_stamp() << ": locked by thread_1, value is " << s.get_value() << std::endl;
            wait(1, SC_SEC); // occupy resource for 1 s
            s.post(); // release resource
            std::cout<< sc_time_stamp() << ": unlocked by thread_1, value is " << s.get_value() << std::endl;
            wait(SC_ZERO_TIME); // give time for the other process to lock
        }
    }
    void thread_2() {
        while (true) {
            if (s.trywait() == -1) { // try to obtain a resource
                s.wait(); // if not successful, wait till resource is available
            }
            std::cout<< sc_time_stamp() << ": locked by thread_2, value is " << s.get_value() << std::endl;
            wait(1, SC_SEC); // occupy resource for 1 s
            s.post(); // release resource
            std::cout<< sc_time_stamp() << ": unlocked by thread_2, value is " << s.get_value() << std::endl;
            wait(SC_ZERO_TIME); // give time for the other process to lock
        }
    }
    void thread_3() {
        while (true) {
            if (s.trywait() == -1) { // try to obtain a resource
                s.wait(); // if not successful, wait till resource is available
            }
            std::cout<< sc_time_stamp() << ": locked by thread_3, value is " << s.get_value() << std::endl;
            wait(1, SC_SEC); // occupy resource for 1 s
            s.post(); // release resource
            std::cout<< sc_time_stamp() << ": unlocked by thread_3, value is " << s.get_value() << std::endl;
            wait(SC_ZERO_TIME); // give time for the other process to lock
        }
    }
};

int sc_main(int, char*[]) {
    SEMAPHORE semaphore("semaphore");
    sc_start(4, SC_SEC);
    return 0;
}