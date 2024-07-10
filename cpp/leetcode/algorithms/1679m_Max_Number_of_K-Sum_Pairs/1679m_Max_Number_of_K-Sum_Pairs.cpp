#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxOperations(vector<int>& nums, int k) 
    {
        int begin = 0, end, count = 0;

        end = nums.size()-1;
        sort(nums.begin(),nums.end());

        while(begin < end)
        {
            if(nums[begin]+nums[end] > k)
                end--;
            else if(nums[begin]+nums[end] < k)
                begin++;
            else //if(nums[begin]+nums[end] == k)
            {
                begin++;
                end--;          
                count++;
            }
            
        }
        return count;
    }
};

void print(vector<int> const &input)
{
    for (int i = 0; i < input.size(); i++)
    {
        cout << input.at(i) << ' ';
    }
    cout << endl;
}

int main()
{
    vector<int> input1 = {1,2,3,4};
    vector<int> input2 = {3,1,3,4,3};

    Solution sol;

    cout << "input 1：" << endl;
    print(input1);
    cout << sol.maxOperations(input1, 5) << endl;
    cout << endl;

    cout << "input 2：" << endl;
    print(input2);
    cout << sol.maxOperations(input2, 6) << endl;

    return 0;
}