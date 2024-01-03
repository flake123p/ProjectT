#pragma once

#include "_list.h"

class TreeClass {
public:
    class TreeClass *prev;
    class TreeClass *left;
    class TreeClass *right;

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

    TreeClass() {
        prev = nullptr;
        left = nullptr;
        right = nullptr;
    }

    /*
        -   -   - / 1
        -   -   3   -
        -   - / - \ 2
        -  10   -   -
        -   -   - / 3
        - / - \ 7   -
        -   -   - \ 4
       36   -   -   -
        -   -   - / 5
        - \ -  11   -
        -   - / - \ 6
        -  26   -   -
        -   - \ - / 7
        -   -  15   -
        -   -   - \ 8
    */
    int dump_children() {
        int max_depth;
        start_traverse(max_depth);
        return max_depth;
    }

    int calc_max_depth() {
        int max_depth;
        start_traverse(max_depth);
        return max_depth;
    }

    void start_traverse(int &max_depth) {
        max_depth = 1;
        int cur_depth = 1;

        if (right != nullptr) {
            right->traverse(max_depth, cur_depth);
        }
        if (left != nullptr) {
            left->traverse(max_depth, cur_depth);
        }
        //printf("max_depth = %d\n", max_depth);
    }

    void traverse(int &max_depth, int &cur_depth) {
        cur_depth++;
        if (cur_depth > max_depth) {
            max_depth = cur_depth;
        }
        if (right != nullptr) {
            right->traverse(max_depth, cur_depth);
        }
        if (left != nullptr) {
            left->traverse(max_depth, cur_depth);
        }
        cur_depth--;
    }
};