#include <vector>
#include <iostream>
using namespace std;

class Solution {
	public:
		void setZeroes(vector<vector<int>>& matrix) {
			const int m = matrix.size();
			const int n = matrix[0].size();
			vector<int> rows(m);
			vector<int> cols(n);
			for (int i=0; i<m; i++) {
				for (int j=0; j<n; j++) {
					rows[i] |= (matrix[i][j]==0);
					cols[j] |= (matrix[i][j]==0);
					// foo |= bar <=> foo = (foo | bar) 
				}
			}
			for (int i=0; i<m; i++) {
				for (int j=0; j<n; j++) {
					if (rows[i] || cols[j])
						matrix[i][j]=0;
				}
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

int main() {
	vector<vector<int> > matrix1;
	vector<vector<int> > matrix2;
	vector<int> r11;
	vector<int> r12;
	vector<int> r13;
	vector<int> r21;
	vector<int> r22;
	vector<int> r23;
	Solution sol;

	r11.push_back(1);
	r11.push_back(1);
	r11.push_back(1);
	r12.push_back(1);
	r12.push_back(0);
	r12.push_back(1);
	r13.push_back(1);
	r13.push_back(1);
	r13.push_back(1);
	matrix1.push_back(r11);
	matrix1.push_back(r12);
	matrix1.push_back(r13);
	DumpMatrix(matrix1);
	sol.setZeroes(matrix1);
	DumpMatrix(matrix1);
	r21.push_back(0);
	r21.push_back(1);
	r21.push_back(2);
	r21.push_back(0);
	r22.push_back(3);
	r22.push_back(4);
	r22.push_back(5);
	r22.push_back(2);
	r23.push_back(1);
	r23.push_back(3);
	r23.push_back(1);
	r23.push_back(5);
	matrix2.push_back(r21);
	matrix2.push_back(r22);
	matrix2.push_back(r23);
	DumpMatrix(matrix2);
	sol.setZeroes(matrix2);
	DumpMatrix(matrix2);
	return 0;
}
