// Learn with Examples, 2020, MIT license

// Backup From: https://www.learnsystemc.com/basic/delta_cycle

/*
A delta cycle can be thought of as a very small step of time within the simulation, which does not increase the user-visible time.
A delta cycle is comprised of separate evaluate and update phases, and multiple delta cycles may occur at a particular simulated time. 
When a signal assignment occurs, other processes do not see the newly assigned value until the next delta cycle.

When is delta cycle used:
  1. notify(SC_ZERO_TIME) causes the event to be notified in the evaluate phase of the next delta cycle, this is called a "delta notification".
  2. A (direct or indirect) call to request_update() causes the update() method to be called in the update phase of the current delta cycle. 
*/

#include <systemc>
using namespace sc_core;

SC_MODULE(DELTA) {
    int x = 1, y = 1; // defines two member variables
    SC_CTOR(DELTA) {
        SC_THREAD(add_x); // x += 2
        SC_THREAD(multiply_x); // x *= 3
        SC_THREAD(add_y); // y += 2
        SC_THREAD(multiply_y); // y *= 3
    }
    void add_x() { // x += 2 happens first
        std::cout << "add_x: " << x << " + 2" << " = ";
        x += 2;
        std::cout << x << std::endl;
        std::cout << "[TimeStamp] " << sc_time_stamp() << std::endl;
    }
    void multiply_x() { // x *= 3 happens after a delta cycle
        wait(SC_ZERO_TIME);
        std::cout << "multiply_x: " << x << " * 3" << " = ";
        x *= 3;
        std::cout << x << std::endl;
        std::cout << "[TimeStamp] " << sc_time_stamp() << std::endl;
    }
    void add_y() { // y += 2 happens after a delta cycle
        wait(SC_ZERO_TIME);
        std::cout << "add_y: " << y << " + 2" << " = ";
        y += 2;
        std::cout << y << std::endl;
        std::cout << "[TimeStamp] " << sc_time_stamp() << std::endl;
    }
    void multiply_y() { // y *=3 happens first
        std::cout << "multiply_y: " << y << " * 3" << " = ";
        y *= 3;
        std::cout << y << std::endl;
        std::cout << "[TimeStamp] " << sc_time_stamp() << std::endl;
    }
};

int sc_main(int, char*[]) {
    DELTA delta("delta");
    sc_start();
    return 0;
}

