#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "Vec.hpp"
using namespace std;

/* Runtims: 25ms, beats 88.94% */
/* Memeory 19.20MB beats 91.03%*/

/* Given an array of strings strs, group the anagrams together. You can return the answer in any order. */

/* 	Input: strs = ["eat","tea","tan","ate","nat","bat"] 
	Output: [["bat"],["nat","tan"],["ate","eat","tea"]]	*/

class Solution {
	public:
		vector<vector<string>> groupAnagrams(vector<string>& strs) {
			vector<vector<string>> result;
			unordered_map<string, vector<int>> hash_table;
			for (int i=0; i<strs.size(); i++){
				string c=strs[i];
				sort(begin(c), end(c));
				hash_table[c].push_back(i);
			}
			for (const auto& kv: hash_table){
				result.push_back({});
				for (int i: kv.second){
					result.back().push_back(strs[i]);
				}
			}
			return result;
		}
};


int main() {
	vector<string> example1;
	vector<string> example2;
	vector<string> example3;
	vector<vector<string> > matrix1_result;
	vector<vector<string> > matrix2_result;
	vector<vector<string> > matrix3_result;
	Solution sol;

	example1.push_back("eat");
	example1.push_back("tea");
	example1.push_back("tan");
	example1.push_back("ate");
	example1.push_back("nat");
	example1.push_back("bat");

	example2.push_back("");

	example3.push_back("a");

	VecDump(example1);
	matrix1_result =sol.groupAnagrams(example1);
	VecDump(matrix1_result);

	VecDump(example2);
	matrix2_result = sol.groupAnagrams(example2);
	VecDump(matrix2_result);

	VecDump(example3);
	matrix3_result = sol.groupAnagrams(example3);
	VecDump(matrix3_result);
	return 0;
}
