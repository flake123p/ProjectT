//
// https://leetcode.com/problems/rotate-image/
//
#include <iostream>
#include <cstdint>
#include <memory>
#include <stdint.h>
#include <vector>

#include "Vec.hpp"

using namespace std;

// Runtime: Beats 100.00%of users with C++
// Memory : Beats 48.67%of users with C++

class Solution {
public:
    void new_pos_clock90(int dim, int &r, int &c) {
        int ori_r = r;
        int ori_c = c;
        c = dim - 1 - ori_r;
        r = ori_c;
        //printf("> %d,%d / %d,%d\n", ori_r, ori_c, r, c);
    }
    void rotate_basic(vector<vector<int>>& matrix, vector<vector<int>>& matrix2) {
        int dim = matrix.size();
        for (int i = 0; i < dim*dim; i++) {
            int r = i % dim;
            int c = i / dim;
            int new_r = r;
            int new_c = c;
            new_pos_clock90(matrix.size(), new_r, new_c);
            matrix2[new_r][new_c] = matrix[r][c];
        }
    }
    void rotate_four_number_swap_clock90(vector<vector<int>>& matrix, int r, int c, int dim) {
        int r0 = r;
        int c0 = c;
        int r1 = r0;
        int c1 = c0;
        new_pos_clock90(dim, r1, c1);
        int r2 = r1;
        int c2 = c1;
        new_pos_clock90(dim, r2, c2);
        int r3 = r2;
        int c3 = c2;
        new_pos_clock90(dim, r3, c3);
        // store 1
        int temp = matrix[r0][c0];
        matrix[r0][c0] = matrix[r3][c3];
        matrix[r3][c3] = matrix[r2][c2];
        matrix[r2][c2] = matrix[r1][c1];
        matrix[r1][c1] = temp;
    }
    void rotate(vector<vector<int>>& matrix) {
        int dim = matrix.size();
        int start_r = 0;
        int start_c = 0;
        // four_number_swap
        // row traverse max, rtm = dim - 1
        // column traverse max, ctm = dim / 2
        int rtm = dim - 1;
        int ctm = dim / 2;

        for (int ct = 0; ct < ctm; ct++) {
            for (int rt = start_r; rt < rtm; rt++) {
                rotate_four_number_swap_clock90(matrix, rt, start_c, dim);
            }
            //next: inner matrix rotate
            start_r += 1;
            start_c += 1;
            rtm -= 1;
        }
    }
};

int main() 
{
    {
        printf("Basic Algorithm:\n");
        int mArray[] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        };
        auto m = VecMake<int>(3, 3, mArray);

        int mxArray[9] = {
            0
        };
        auto mx = VecMake<int>(3, 3, mxArray);

        VecDump(m);
        VecDump(mx);

        Solution sol;
        sol.rotate_basic(m, mx);

        VecDump(mx);
    }

    {
        printf("Complex Algorithm:\n");
        int mArray[] = {
             5,  1,  9, 11,
             2,  4,  8, 10,
            13,  3,  6,  7,
            15, 14, 12, 16,
        };
        auto m = VecMake<int>(4, 4, mArray);
        VecDump(m);
        Solution sol;
        sol.rotate(m);
        VecDump(m);
    }

    return 0;
}
