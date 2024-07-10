#pragma once

#include "math.h"
#include "_list.h"
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

    enum class DrawType : int {
        Null = 0, //default
        Node,
        Root,
        Slash,
        Backslash,
    };
    struct drawInfo {
        DrawType type;
        info *pInfo;
    };

    int drawLink_ = 1;
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
    template<typename ChildFunc_t> // [](class TreeClass *prev, int isLeft) -> class TreeClass * {
    int TreeNodesRegister(T root, ChildFunc_t f) {

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
                    int isLeft;
                    if (curNodeIdx % 2 == 1) {
                        isLeft = 0;
                    } else {
                        isLeft = 1;
                    }
                    array2D_->ref({r, c}).node = f(prevInfo->node, isLeft);
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

    info &Info(const std::initializer_list<int>& indices) {
        return array2D_->ref(indices);
    };

    int Draw() {
        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                if (Info({r, c}).node != nullptr) {
                    printf("[%3d] ", Info({r, c}).index);
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
    template<typename TUIFunc_t> // [](class TreeClass *node) -> void {
    int DrawV(TUIFunc_t f) {
        BASIC_ASSERT(array2D_->shape_.size() == 2);
        int row = array2D_->shape_[0];
        int col = array2D_->shape_[1];

        int layoutRows = (row * 2);
        int layoutCols = (col * 2);
        int elemNumInCol;
        int rowNumPerElem;
        int rowPaddingPerElem;
        int newRol;
        int newCol;

        class ArTen<struct drawInfo> verticalLayout({layoutRows, layoutCols});

        // Remap row index to vertical layout
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                if (Info({r, c}).node != nullptr) {
                    elemNumInCol = pow(2, c);
                    rowNumPerElem = layoutRows / elemNumInCol;
                    rowPaddingPerElem = rowNumPerElem / 2;
                    BASIC_ASSERT(rowPaddingPerElem > 0); //must > 0, because first line must be empty

                    newRol = (r * rowNumPerElem) + rowPaddingPerElem;
                    newCol = (c * 2) + 1;
                    //printf("> %d,%d,%d,%d\n", newRol, r, rowNumPerElem, rowPaddingPerElem);
                    BASIC_ASSERT(newRol < layoutRows);
                    verticalLayout.ref({newRol, newCol}).pInfo = &Info({r, c});
                    verticalLayout.ref({newRol, newCol}).type = DrawType::Node;
                    //
                    // Add link
                    //
                    DrawType newType;
                    int rowOffset = rowPaddingPerElem / 2;
                    // root
                    if (Info({r, c}).index == 0) {
                        rowOffset = 0;
                        newType = DrawType::Root;
                    }
                    // right
                    if ((Info({r, c}).index % 2) == 1) {
                        newType = DrawType::Slash;
                    }
                    // left
                    else {
                        rowOffset = rowOffset * -1;
                        newType = DrawType::Backslash;
                    }
                    newRol = newRol + rowOffset;
                    newCol = newCol - 1;
                    verticalLayout.ref({newRol, newCol}).type = newType;
                }
            }
        }

        if (drawLink_) {
            for (int r = 1; r < layoutRows; r++) {
                for (int c = 1; c < layoutCols; c+=1) {
                    if (verticalLayout.ref({r, c}).pInfo != nullptr) {
                        printf("[%2d] ", verticalLayout.ref({r, c}).pInfo->index);
                        f(verticalLayout.ref({r, c}).pInfo->node);
                    } 
                    else {
                        if ((c % 2) == 0) {
                            // Link Columns
                            if (verticalLayout.ref({r, c}).type == DrawType::Slash) {
                                printf(" / ");
                            }
                            else if (verticalLayout.ref({r, c}).type == DrawType::Backslash) {
                                printf(" \\ ");
                            } 
                            else {
                                printf("   ");
                            }
                        }
                        else {
                            printf(" --- ");
                            f(nullptr);
                        }
                    }
                }
                printf("\n");
            }
        } else {
            for (int r = 1; r < layoutRows; r++) {
                for (int c = 1; c < layoutCols; c+=2) {
                    if (verticalLayout.ref({r, c}).pInfo != nullptr) {
                        printf("[%3d] ", verticalLayout.ref({r, c}).pInfo->index);
                    } else {
                        printf(" ---  ");
                    }
                }
                printf("\n");
            }
        }
        //printf("\n");
        return 0;
    }
};
