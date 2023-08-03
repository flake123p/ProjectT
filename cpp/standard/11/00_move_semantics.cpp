
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

*/

int buf0[2] = {1, 2};
int buf1[2] = {3, 9};

class DemoBasic {
public:
    int i;
    int *p;
    DemoBasic(int in_i=9999){
        i = in_i;
        p = NULL;
        printf("construct DemoBasic - %d, %p\n", i, p);};
    ~DemoBasic(){printf("destruct DemoBasic - %d, %p\n", i, p);};
};

void object_move()
{
    printf("%s() --START--\n", __func__);

    DemoBasic obj0(777);
    DemoBasic obj1(999);

    obj0.p = buf0;
    obj1.p = buf1;

    printf("buf0 = %d, %d\n", buf0[0], buf0[1]);
    printf("buf1 = %d, %d\n", buf1[0], buf1[1]);
    printf("obj0 = %d, %d, %d\n", obj0.i, obj0.p[0], obj0.p[1]);
    printf("obj1 = %d, %d, %d\n", obj1.i, obj1.p[0], obj1.p[1]);

    obj0 = std::move(obj1);
    printf("after move() ...\n");

    printf("buf0 = %d, %d\n", buf0[0], buf0[1]);
    printf("buf1 = %d, %d\n", buf1[0], buf1[1]);
    printf("obj0 = %d, %d, %d\n", obj0.i, obj0.p[0], obj0.p[1]);
    printf("obj1 = %d, %d, %d\n", obj1.i, obj1.p[0], obj1.p[1]);

    printf("%s() --END--\n", __func__);
}


std::vector<int> vector_move_func()
{
    std::vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(4);
    return std::move(vec);
}

void vector_move()
{
    printf("%s() --START--\n", __func__);

    std::vector<int> vec0 = vector_move_func();
    printf("vec0 size = %lu\n", vec0.size());

    printf("%s() --END--\n", __func__);
}

/*
    Flake: still 2 object

    move = pointer clone https://charlottehong.blogspot.com/2017/03/stdmove.html

    cppreference : https://en.cppreference.com/w/cpp/utility/move

*/
int main()
{
    object_move();

    vector_move();

    return 0;
}
