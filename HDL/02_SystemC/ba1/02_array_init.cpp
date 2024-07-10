

#include <stdint.h>
#include <string>
#include <systemc>
using namespace sc_core;

typedef void (*task_cb_t)(int idx);

SC_MODULE(CONCURRENCY) {
    SC_CTOR(CONCURRENCY) { // constructor
        SC_THREAD(thread1); // register thread1
    }
    void thread1() {
        while(true) { // infinite loop
            std::cout << sc_time_stamp() << ": thread1" << std::endl;
            if (cb != nullptr) {
                cb(idx);
            }
            wait(1, SC_SEC); // trigger again after 2 "simulated" seconds
        }
    }
    task_cb_t cb = nullptr;
    int idx = 0;
};

void t0(int idx) {
    printf("%s() idx=%d\n", __func__, idx);
}

template<typename T0>
void sc_array_init(T0 *vec[], int num, std::string prefix)
{
    std::string new_str;

    for (int i = 0; i < num; i++) {
        new_str = prefix + std::to_string(i);
        vec[i] = new T0(new_str.c_str());
    }
}

template<typename T0>
void sc_array_uninit(T0 *vec[], int num)
{
    for (int i = 0; i < num; i++) {
        delete vec[i];
    }
}

CONCURRENCY *vec[10];
int sc_main(int, char*[]) {

    sc_array_init(vec, 4, "abc");

    for (int i = 0; i < 3; i++) {
        vec[i]->cb = t0;
        vec[i]->idx = i;
    }

    sc_start(5, SC_SEC); // run simulation for 10 seconds

    sc_array_uninit(vec, 4);
    return 0;
}
