#include <cstdlib>
//
//
//
template<typename T>
struct AryObject {
    T *buf;
    int num;
    AryObject(int input_num) {
        num = input_num;
        buf = new T[num];
    }
    ~AryObject() {
        delete[] buf;
    }

    T& operator[](int idx)
    {
        return buf[idx];
    }
};

template<typename T>
struct AryEmpty {
    T *buf = nullptr;
    int num;
    AryEmpty(int input_num) {
        num = input_num;
        buf = (T *)calloc(num, sizeof(T));
    }
    ~AryEmpty() {
        if (buf != nullptr) {
            free(buf);
            buf = nullptr;
        }
    }

    T& operator[](int idx)
    {
        return buf[idx];
    }
};


int num = 3;

// int x[num];
AryObject<int> x(num);
AryEmpty<int> y(num);

int main()
{
    x[2] = 99;
    return x[2] + y[2];
}