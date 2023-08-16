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

// Runtime: Beats 30.58%of users with C++
// Memory : Beats 52.56%of users with C++

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ret;
        int max = nums.size();
        for (int i = 0; i < max; i++) {
            for (int j = i+1; j < max; j++) {
                if (nums[i] + nums[j] == target) {
                    ret.push_back(i);
                    ret.push_back(j);
                    return std::move(ret);
                }
            }
        }
        return std::move(ret);
    }
};

int main() 
{
    {
        int target = 9;
        int inputArray[] = {2, 7, 11, 15};
        auto input = VecMake<int>(LEN(inputArray), inputArray);
        
        Solution sol;
        auto ans = sol.twoSum(input, target);
        VecDump(ans);
    }
    {
        int target = 6;
        int inputArray[] = {3, 2, 4};
        auto input = VecMake<int>(LEN(inputArray), inputArray);
        
        Solution sol;
        auto ans = sol.twoSum(input, target);
        VecDump(ans);
    }
    {
        int target = 6;
        int inputArray[] = {3, 3};
        auto input = VecMake<int>(LEN(inputArray), inputArray);
        
        Solution sol;
        auto ans = sol.twoSum(input, target);
        VecDump(ans);
    }
    return 0;
}
