// Learn with Examples, 2020, MIT license

// https://www.learnsystemc.com/basic/concurrency

#include <systemc>
#include <stdint.h>
using namespace sc_core;

uint32_t status_r[2] = {0};
uint32_t status_w[2] = {0};

int request_check(int index)
{
    if (status_r[index] != status_w[index]) {
        status_w[index]++;
        return 1;
    } else {
        return 0;
    }
}

void trigger(int index) {
    wait(SC_ZERO_TIME);
    status_r[index]++;
    wait(1, SC_SEC);
}

typedef void (*cb_t)(void);

void t0() {
    // printf("%s\n", __func__);

    if (request_check(0)) {
        printf("msg to t0!!\n");
        trigger(1);
    }
}

void t1() {
    // printf("%s\n", __func__);

    if (request_check(1)) {
        printf("msg to t1!!\n");
        trigger(0);
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

    status_r[0] = 1;

    sc_start(10, SC_SEC); // run simulation for 10 seconds
    return 0;
}
