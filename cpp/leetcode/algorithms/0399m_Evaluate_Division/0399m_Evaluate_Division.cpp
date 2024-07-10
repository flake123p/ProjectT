#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "Vec.hpp"
using namespace std;

/* Runtims: 0ms, beats 100.0% */
/* Memeory 9.16MB beats 12.66%*/

/* Pls visit Leetcode@399 */
/* Graph + DPS */

class Solution {
	public:
		vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
			vector<double> res;
			// g[A][B] aka A/B = value
			unordered_map<string, unordered_map<string, double>> graph;
			for (int i = 0; i < equations.size(); ++i){
				string A = equations[i][0];
				string B = equations[i][1];
				double K = values[i];
				graph[A][B] = K;
				graph[B][A] = 1/K;
			}
			for (const auto pair: queries){
				string X = pair[0];
				string Y = pair[1];
				if (!graph.count(X)||!graph.count(Y)){
					res.push_back(-1);
					continue;
				}
				unordered_set<string> visited;
				res.push_back(divide(X, Y, graph, visited));
			}
			return res;
		}
	private:
		double divide(string A, string B, unordered_map<string, unordered_map<string, double>>& graph, unordered_set<string>& visited){
			if (A==B){
			       	return 1.0;
			}
			visited.insert(A);
			for (const auto pair: graph[A]){
				string C = pair.first;
				if (visited.count(C)){
					continue;
				}
				double d = divide(C, B, graph, visited);
				if (d>0) {
					return d*graph[A][C];
				}
			}
			return -1;
		}
};


int main() {
	vector<vector<string>> example1;
	vector<double> value1;
	vector<vector<string>> queries1;
	vector<double> res1;
	vector<vector<string>> example2;
	vector<double> value2;
	vector<vector<string>> queries2;
	vector<double> res2;
	Solution sol;

	example1.push_back({"a", "b"});
	example1.push_back({"b", "c"});
	value1.push_back(2);
	value1.push_back(3);
	queries1.push_back({"a", "c"});
	queries1.push_back({"b", "a"});
	queries1.push_back({"a", "e"});
	queries1.push_back({"a", "a"});
	queries1.push_back({"x", "x"});
	
	example2.push_back({"a", "b"});
	example2.push_back({"b", "c"});
	example2.push_back({"bc", "cd"});
	value2.push_back(1.5);
	value2.push_back(2.5);
	value2.push_back(5.0);
	queries2.push_back({"a", "c"});
	queries2.push_back({"c", "b"});
	queries2.push_back({"bc", "cd"});
	queries2.push_back({"cd", "bc"});

	res1 = sol.calcEquation(example1, value1, queries1);
	VecDump(res1);
	res2 = sol.calcEquation(example2, value2, queries2);
	VecDump(res2);
	return 0;
}
