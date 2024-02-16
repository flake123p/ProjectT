//
// https://cplusplus.com/reference/mutex/lock_guard/
//

// lock_guard example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::lock_guard
#include <stdexcept>      // std::logic_error
#include <vector>

#error NOT FINISH


template<typename T>
class Event {
public:
    std::mutex mtxSend;
    std::mutex mtxRecv;
    T val;
    int inited = 0;

    Event() {
        // int x;
        // x = mtx.try_lock();
        // printf("S e0 try_lock = %d\n", x);
        // x = mtx.try_lock();
        // printf("S e0 try_lock = %d\n", x);

        mtxRecv.lock();
        mtxSend.lock();
    };

    void WaitLock() {
        int x;

        while (1) {
            x = mtxRecv.try_lock();
            if (x) {
                mtxRecv.unlock();
            } else {
                break;
            }
        }
    }

    void RecvStart() {
        mtxSend.unlock();
        mtxRecv.lock();
    }

    void RecvEnd() {
        mtxRecv.unlock();
        // mtxRecv.lock();
    }

    void SendStart() {
        WaitLock();
        // int x = mtx.try_lock();
        // printf("try_lock = %d\n", x);
        mtxSend.lock();
    }

    void SendEnd() {
        mtx.unlock();
        // mtx.lock();
    }
};

template<typename T>
class Events {
public:
    std::vector<class Event<T> *> vec;
    // class Event evt;
    // void *pContent;

    Events(int num) {
        vec.clear();
        for (int i = 0; i < num; i++) {
            auto x = new class Event<T>;
            vec.push_back(x);
        }
    }

    ~Events() {
        for (auto it = vec.begin() ; it != vec.end(); ++it) {
            delete(*it);
        }
    }

    void WaitLockAll() {
        for (auto it = vec.begin() ; it != vec.end(); ++it) {
            (*it)->WaitLock();
        }
    }

    void WaitInitDone() {
        while (1) {
            int rewind = 0;
            for (auto it = vec.begin() ; it != vec.end(); ++it) {
                if ((*it)->inited == 0) {
                    rewind = 1;
                    break;
                }
            }
            if (rewind) {
                continue;
            } else {
                break;
            }
        }
    }

    class Event<T> *operator[] (int idx) {
        return vec[idx];
    }
};

//class Event<int> evt[3];
class Events<int> evt(3);

void job (int id) {
    printf("id=%d\n", id);
    evt[id]->inited = 1;

    while (1) {
        evt[id]->RecvStart();
        printf("id=%d, val=%d\n", id, evt[id]->val);
        evt[id]->RecvEnd();
    }

}

int main ()
{

    std::thread threads[3];
    // spawn 10 threads:
    for (int i=0; i<3; ++i) {
        threads[i] = std::thread(job, i);
    }

    printf("xxx\n");
    evt.WaitInitDone();
    printf("yyy\n");

    evt[1]->SendStart();
    evt[1]->val = 2244;
    evt[1]->SendEnd();

    evt[1]->SendStart();
    evt[1]->val = 6688;
    evt[1]->SendEnd();

    // for (int i = 0; i < 10; i++) {
    //     int x = evt[0].mtx.try_lock();
    //     printf("E e0 try_lock = %d\n", x);
    // }

    for (auto& th : threads) {
        th.join();
    }

    return 0;
}