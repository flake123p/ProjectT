// Learn with Examples, 2020, MIT license

// Backup from: https://www.learnsystemc.com/basic/channel_fifo

/*
sc_fifo:
  1. is a predefined primitive channel intended to model the behavior of a fifo, i.e., a first in first out buffer.
  2. has a number of slots for storing values. The number of slots is fixed when the object is constructed.
  3. implements the sc_fifo_in_if<T> interface and the sc_fifo_out_if<T> interface.

Constructors:
  1. explicit sc_fifo(int size_ = 16): calls the base class constructor from its initializer list as: sc_prim_channel(sc_gen_unique_name( "fifo" ))
  2. explicit sc_fifo(const char* name_, int size_ = 16): calls the base class constructor from its initializer list as: sc_prim_channel(name_)
  Both constructors initialize the number of slots in the fifo to the value given by the parameter size_. The number of slots shall be greater than zero.

Member functions for read:
  1. void read(T&), T read():
    a) return the value least recently written into the fifo and remove that value from the fifo such that it cannot be read again.
    b) the order in which values are read from the fifo shall precisely match the order in which values were written into the fifo.
    c) values written into the fifo during the current delta cycle are not available for reading in that delta cycle, but become available for reading in the immediately following delta cycle.
    d) if the fifo is empty, shall suspend until the data-written event is notified.
  2. bool nb_read(T&):
    a), b), c) same as read()
    d) if the fifo is empty, member function nb_read shall return immediately without modifying the state of the fifo, without calling request_update, and with a return value of false. Otherwise, if a value is available for reading, the return value of nb_read() shall be true.
  3. operator T(): equivalent to "operator T() {return read();}""

Member functions for write:
  1. write(const T&):
    a) write the value passed as an argument into the fifo.
    b) multiple values may be written within a single delta cycle.
    c) if values are read from the fifo during the current delta cycle, the empty slots in the fifo so created do not become free for the purposes of writing until the immediately following delta cycle.
    d) if the fifo is full, write() shall suspend until the data-read event is notified.
  2. bool nb_write(const T&):
    a), b), c) same as write()
    d) if the fifo is full, nb_write() shall return immediately without modifying the state of the fifo, without calling request_update, and with a return value of false. Otherwise, the return value of nb_write() shall be true.
  3. operator=: equivalent to "sc_fifo<T>& operator= (const T& a) {write(a); return *this;}"

Member functions for events:
  1. sc_event& data_written_event(): shall return a reference to the data-written event, that is notified in the delta notification phase that occurs at the end of the delta cycle in which a value is written into the fifo.
  2. sc_event& data_read_event(): shall return a reference to the data-read event, that is notified in the delta notification phase that occurs at the end of the delta cycle in which a value is read from the fifo.

Member functions for available values and free slots:
  1. int num_available(): returns the number of values that are available for reading in the current delta cycle. The calculation shall deduct any values read during the current delta cycle but shall not add any values written during the current delta cycle.
  2. int num_free(): returns the number of empty slots that are free for writing in the current delta cycle. The calculation shall deduct any slots written during the current delta cycle but shall not add any slots made free by reading in the current delta cycle.
*/
#include <systemc>
using namespace sc_core;

SC_MODULE(FIFO) {
    sc_fifo<int> f1, f2, f3;
    SC_CTOR(FIFO) : f1(2), f2(2), f3(2) { // fifo with size 2
        SC_THREAD(generator1);
        SC_THREAD(consumer1);

        SC_THREAD(generator2);
        SC_THREAD(consumer2);

        SC_THREAD(generator3);
        SC_THREAD(consumer3);
    }
    void generator1() { // blocking write
        int v = 0;
        while (true) {
            f1.write(v); // same as f = v, which is not recommended.
            std::cout << sc_time_stamp() << ": generator1 writes " << v++ << std::endl;
            wait(1, SC_SEC); // write every 1 s
        }
    }
    void consumer1() { // blocking read
        int v = -1;
        while (true) {
            f1.read(v); // same as v = int(f), which is not recommended; or, v = f1.read();
            std::cout << sc_time_stamp() << ": consumer1 reads " << v << std::endl;
            wait(3, SC_SEC); // read every 3 s, fifo will fill up soon
        }
    }
    void generator2() { // non-blocking write
        int v = 0;
        while (true) {
            while (f2.nb_write(v) == false ) { // nb write until succeeded
                wait(f2.data_read_event()); // if not successful, wait for data read (a fifo slot becomes available)
            }
            std::cout << sc_time_stamp() << ": generator2 writes " << v++ << std::endl;
            wait(1, SC_SEC); // write every 1 s
        }
    }
    void consumer2() { // non-blocking read
        int v = -1;
        while (true) {
            while (f2.nb_read(v) == false) {
                wait(f2.data_written_event());
            }
            std::cout << sc_time_stamp() << ": consumer2 reads " << v << std::endl;
            wait(3, SC_SEC); // read every 3 s, fifo will fill up soon
        }
    }
    void generator3() { // free/available slots before/after write
        int v = 20;
        while (true) {
            std::cout << sc_time_stamp() << ": generator3, before write, #free/#available = " << f3.num_free() << "/" << f3.num_available() << ", v = " << v << std::endl;
            f3.write(v++);
            std::cout << sc_time_stamp() << ": generator3, after write, #free/#available = " << f3.num_free() << "/" << f3.num_available() << ", v = " << v << std::endl;
            wait(1, SC_SEC);
        }
    }
    void consumer3() { // free/available slots before/after read
        int v = -1;
        while (true) {
            std::cout << sc_time_stamp() << ": consumer3, before read, #free/#available = " << f3.num_free() << "/" << f3.num_available() << ", v = " << v << std::endl;
            f3.read(v);
            std::cout << sc_time_stamp() << ": consumer3, after read, #free/#available = " << f3.num_free() << "/" << f3.num_available() << ", v = " << v << std::endl;
            wait(3, SC_SEC); // read every 3 s, fifo will fill up soon
        }
    }
};

int sc_main(int, char*[]) {
    FIFO fifo("fifo");
    sc_start(16, SC_SEC);
    return 0;
}
