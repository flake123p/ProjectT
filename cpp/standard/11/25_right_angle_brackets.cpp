
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    C++11 is now able to infer when a series of right angle brackets is used as an 
    operator or as a closing statement of typedef, without having to add whitespace.
*/

typedef std::map<int, std::map <int, std::map <int, int> > > cpp98LongTypedef;
typedef std::map<int, std::map <int, std::map <int, int>>>   cpp11LongTypedef;

int main()
{
  {
    cpp98LongTypedef cpp98;
    cpp11LongTypedef cpp11;
  }

  return 0;
}
