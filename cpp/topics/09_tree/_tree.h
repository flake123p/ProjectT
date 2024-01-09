#pragma once

#include "math.h"
#include "_list.h"

class TreeClass {
public:
    class TreeClass *prev;
    class TreeClass *left;
    class TreeClass *right;
    int autoFree;
    void *payload = nullptr;

    void set_prev(class TreeClass &input)
    {
        prev = &input;
    }

    void set_right(class TreeClass &input)
    {
        right = &input;
        input.set_prev(*this);
    }

    void set_left(class TreeClass &input)
    {
        left = &input;
        input.set_prev(*this);
    }

    void show() {} //overwrite this for dump ...

    TreeClass(int autoFree = 0) {
        prev = nullptr;
        left = nullptr;
        right = nullptr;
        this->autoFree = autoFree;
    }

    int calc_max_depth() {
        int max_depth = traverse(
            [](class TreeClass *cur, int cur_depth) -> void {
                cur = cur; //dummy
                cur_depth = cur_depth; //dummy
            }
        );
        return max_depth;
    }

    template<typename TravFunc_t> // [](class TreeClass *cur, int cur_depth) -> void {
    int traverse(TravFunc_t f) {
        int max_depth = 0;
        int cur_depth = 0;

        f(this, cur_depth);

        if (right != nullptr) {
            right->traverse_depth(max_depth, cur_depth, f);
        }
        if (left != nullptr) {
            left->traverse_depth(max_depth, cur_depth, f);
        }
        //printf("max_depth = %d\n", max_depth);
        return max_depth;
    }
    template<typename TravFunc_t>
    void traverse_depth(int &max_depth, int &cur_depth, TravFunc_t f) {
        cur_depth++;
        if (cur_depth > max_depth) {
            max_depth = cur_depth;
        }

        f(this, cur_depth);

        if (right != nullptr) {
            right->traverse_depth(max_depth, cur_depth, f);
        }
        if (left != nullptr) {
            left->traverse_depth(max_depth, cur_depth, f);
        }
        cur_depth--;
    }
};
