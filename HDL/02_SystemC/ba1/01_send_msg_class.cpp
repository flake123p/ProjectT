// Learn with Examples, 2020, MIT license

// https://www.learnsystemc.com/basic/concurrency

#include <systemc>
#include <stdint.h>
using namespace sc_core;

class Trigger {
public:
    uint32_t r_index;
    uint32_t w_index;
    Trigger() : r_index(0), w_index(0) {};
    
    int check()
    {
        if (r_index != w_index) {
            w_index++;
            return 1;
        } else {
            return 0;
        }
    }

    void toggle(int do_wait = 1)
    {
        if (do_wait) {
            wait(SC_ZERO_TIME);
        }
        r_index++;
        if (do_wait) {
            wait(1, SC_SEC);
        }
    }
};

class Trigger g_TaskFlag[2];
class Trigger g_MemFlag[2];

typedef void (*cb_t)(void);

void t0() {
    // printf("%s\n", __func__);

    if (g_TaskFlag[0].check()) {
        printf("msg to t0!!\n");
        g_TaskFlag[1].toggle();
    }
}

void t1() {
    // printf("%s\n", __func__);

    if (g_TaskFlag[1].check()) {
        printf("msg to t1!!\n");
        g_TaskFlag[0].toggle();
    }
}

SC_MODULE(CONCURRENCY) {
    SC_CTOR(CONCURRENCY) { // constructor
        SC_THREAD(thread1); // register thread1
    }
    void thread1() {
        while(true) { // infinite loop
            std::cout << sc_time_stamp() << ": thread1" << std::endl;
            if (cb != nullptr) {
                cb();
            }
            wait(1, SC_SEC); // trigger again after 2 "simulated" seconds
        }
    }
    cb_t cb = nullptr;
};

int sc_main(int, char*[]) {
    CONCURRENCY r0("r0"); // define an object
    r0.cb = t0;
    CONCURRENCY r1("r1"); // define an object
    r1.cb = t1;

    g_TaskFlag[0].toggle(0);

    sc_start(10, SC_SEC); // run simulation for 10 seconds
    return 0;
}
