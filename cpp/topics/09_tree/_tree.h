#pragma once

#include "math.h"
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

    int calc_max_depth() {
        int max_depth = 0;
        int cur_depth = 0;

        if (right != nullptr) {
            right->traverse_depth(max_depth, cur_depth);
        }
        if (left != nullptr) {
            left->traverse_depth(max_depth, cur_depth);
        }
        //printf("max_depth = %d\n", max_depth);
        return max_depth;
    }

    void traverse_depth(int &max_depth, int &cur_depth) {
        cur_depth++;
        if (cur_depth > max_depth) {
            max_depth = cur_depth;
        }
        if (right != nullptr) {
            right->traverse_depth(max_depth, cur_depth);
        }
        if (left != nullptr) {
            left->traverse_depth(max_depth, cur_depth);
        }
        cur_depth--;
    }
};

#include "_arTen.h"

template<typename T>
class TreeDraw {
public:
    // Complete Binary Tree
    struct cbtInfo {
        // All members has initialized with calloc() in ArTen constructor.
        int index;
        T node;
        struct cbtInfo *prevInfo;
        struct cbtInfo *rightInfo; //right child
        struct cbtInfo *leftInfo; //left child
    };
    using info = struct cbtInfo;

    int treeDepth_; // +1 = column, start from 0
    class ArTen<info> *array2D_;

    TreeDraw(int treeDepth) {
        BASIC_ASSERT(treeDepth >= 0);

        treeDepth_ = treeDepth;

        int ary_height = (int)pow(2, treeDepth_); // row

        //printf("%d, %d\n", treeDepth_, ary_height);
        array2D_ = new class ArTen<info>({ary_height, treeDepth_ + 1});

        //array2D_->dump();

        SetAllInfo();
    }
    ~TreeDraw() {
        delete(array2D_);
    }

    void DumpWithIndice() {
        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                printf("%3d, ", array2D_->ref({r, c}).index);
            }
            printf("\n");
        }
        printf("\n");
    }

    //
    // Checkout the illustration below.
    //
    int SetAllInfo() {
        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];

        int curNodeIdx = 0;/*root*/
        for (int curCol = 0; curCol < col; curCol++) {
            int maxNodesThisLevel = (int)pow(2, curCol);
            int nodesCtr = 0;
            for (int curRow = 0; curRow < row; curRow++) {
                // Set index
                array2D_->ref({curRow, curCol}).index = curNodeIdx;

                // Set previous to next column
                if (curCol < col - 1) {
                    int nextCol = curCol + 1;
                    int nextColStartIdx = (int)pow(2, nextCol) - 1;
                    int rRow = ((curNodeIdx * 2) + 1) - nextColStartIdx;
                    int lRow = ((curNodeIdx * 2) + 2) - nextColStartIdx;

                    //printf("[%d] %d,%d,%d,%d,%d\n", __LINE__, rRow, lRow, nextCol, curRow, curCol);
                    array2D_->ref({rRow, nextCol}).prevInfo = &array2D_->ref({curRow, curCol});
                    array2D_->ref({lRow, nextCol}).prevInfo = &array2D_->ref({curRow, curCol});

                    array2D_->ref({curRow, curCol}).rightInfo = &array2D_->ref({rRow, nextCol});
                    array2D_->ref({curRow, curCol}).leftInfo = &array2D_->ref({lRow, nextCol});
                }

                curNodeIdx++;
                nodesCtr++;
                if (nodesCtr >= maxNodesThisLevel) {
                    BASIC_ASSERT(nodesCtr == maxNodesThisLevel);
                    break;
                }
            }
        }
        return 0;
    }
    //
    // Please register in breadth first order!!! (Must register upper part of tree, then lower part.)
    // The index of CBT(Complete Binary Tree) is in right node first order.
    //
    // Depth    Root            => 2D array
    //  0        0                  0  1  3  7
    //  1      2   1                   2  4  ...
    //  2     6 5 4 3                     5  ...
    //  3            7                    6  ...
    //                                       ...
    //                                       ...
    //                                       ...
    //                                       ...
    //
    template<typename rFunc_t, typename lFunc_t>
    int TreeNodesRegister(T root, rFunc_t rFunc, lFunc_t lFunc) {

        array2D_->ref({0, 0}).node = root;
        // array2D_->ref({0, 1}).node = rFunc(root);
        // array2D_->ref({1, 1}).node = lFunc(root);

        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];
        struct cbtInfo *prevInfo;

        int curNodeIdx = 1;/*skip root*/
        for (int c = 1/*skip root*/; c < col; c++) {
            int maxNodesThisLevel = (int)pow(2, c);
            int nodesCtr = 0;
            for (int r = 0; r < row; r++) {
                prevInfo = array2D_->ref({r, c}).prevInfo;

                //printf("[%d] curNodeIdx = %d\n", __LINE__, curNodeIdx);
                BASIC_ASSERT(prevInfo != nullptr);
                
                if (prevInfo->node != nullptr) {
                    //printf("[%d] prev index = %d\n", __LINE__, prevInfo->index);
                    if (curNodeIdx % 2 == 1) {
                        array2D_->ref({r, c}).node = rFunc(prevInfo->node);
                    } else {
                        array2D_->ref({r, c}).node = lFunc(prevInfo->node);
                    }
                }

                curNodeIdx++;
                nodesCtr++;
                if (nodesCtr >= maxNodesThisLevel) {
                    BASIC_ASSERT(nodesCtr == maxNodesThisLevel);
                    break;
                }
            }
        }
        return 0;
    }

    info &ref(const std::initializer_list<int>& indices) {
        return array2D_->ref(indices);
    };

    int Draw() {
        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                if (ref({r, c}).node != nullptr) {
                    printf("[%3d] ", ref({r, c}).index);
                } else {
                    printf("      ");
                }
            }
            printf("\n");
        }
        //printf("\n");
        return 0;
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
    int DrawV() {
        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];

        int totalLines = (row * 2) - 0;
        int elemNumInCol;
        int paddingInCol;
        int premablesInCol;
        int newRol;

        class ArTen<info *> verticalLayout({totalLines, col});

        // Remap row index to vertical layout
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                if (ref({r, c}).node != nullptr) {
                    elemNumInCol = pow(2, c);
                    paddingInCol = totalLines / elemNumInCol;
                    premablesInCol = paddingInCol / 2;

                    newRol = (r * paddingInCol) + premablesInCol;
                    //printf("> %d,%d,%d,%d\n", newRol, r, paddingInCol, premablesInCol);
                    BASIC_ASSERT(newRol < totalLines);
                    verticalLayout.ref({newRol, c}) = &ref({r, c});
                }
            }
        }

        for (int r = 0; r < totalLines; r++) {
            for (int c = 0; c < col; c++) {
                if (verticalLayout.ref({r, c}) != nullptr) {
                    printf("[%3d] ", verticalLayout.ref({r, c})->index);
                } else {
                    printf(" ---  ");
                }
            }
            printf("\n");
        }
        //printf("\n");
        return 0;
    }
};