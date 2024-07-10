
#include <iostream>
#include <cstdint>
#include <memory>
#include <stdint.h>
#include <vector>
using namespace std;

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

void DumpMatrix(vector<vector<int>> &m) {
    printf("%s():\n", __func__);
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            printf("%4d ", m[i][j]);
        }
        printf("\n");
    }
}

int main() 
{
    vector<vector<int>> m;
    vector<int> r0;
    vector<int> r1;
    vector<int> r2;
    vector<int> r3;
    
    r0.push_back(1);
    r0.push_back(2);
    r0.push_back(3);
    m.push_back(r0);

    r1.push_back(4);
    r1.push_back(5);
    r1.push_back(6);
    m.push_back(r1);
    
    r2.push_back(7);
    r2.push_back(8);
    r2.push_back(9);
    m.push_back(r2);

    vector<vector<int>> mX;
    vector<int> r0X;
    vector<int> r1X;
    vector<int> r2X;
    
    r0X.push_back(0);
    r0X.push_back(0);
    r0X.push_back(0);
    mX.push_back(r0X);

    r1X.push_back(0);
    r1X.push_back(0);
    r1X.push_back(0);
    mX.push_back(r1X);
    
    r2X.push_back(0);
    r2X.push_back(0);
    r2X.push_back(0);
    mX.push_back(r2X);

    DumpMatrix(m);
    DumpMatrix(mX);

    Solution sol;
    sol.rotate_basic(m, mX);

    DumpMatrix(mX);

    printf("Complex Algorithm:\n");
    m.clear();
    r0.clear();
    r1.clear();
    r2.clear();
    r3.clear();

    r0.push_back(5);  r0.push_back(1);  r0.push_back(9);  r0.push_back(11); m.push_back(r0);
    r1.push_back(2);  r1.push_back(4);  r1.push_back(8);  r1.push_back(10); m.push_back(r1);
    r2.push_back(13); r2.push_back(3);  r2.push_back(6);  r2.push_back(7);  m.push_back(r2);
    r3.push_back(15); r3.push_back(14); r3.push_back(12); r3.push_back(16); m.push_back(r3);
    DumpMatrix(m);
    sol.rotate(m);
    DumpMatrix(m);

    return 0;
}
