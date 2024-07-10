
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>

/*
    https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

    All members of an inline namespace are treated as if they were part of its parent namespace, 
    allowing specialization of functions and easing the process of versioning. 
    
    This is a transitive property, 
    if A contains B, which in turn contains C and both B and C are inline namespaces, 
    C's members can be used as if they were on A.
*/

namespace Program {
  namespace Version1 {
    int getVersion() { return 1; }
    bool isFirstVersion() { return true; }
  }
  inline namespace Version2 {
    int getVersion() { return 2; }
  }
  namespace Version3 {
    int getVersion() { return 3; }
    //bool isFirstVersion() { return true; }
  }
}

int main()
{
  {
    int version {Program::getVersion()};              // Uses getVersion() from Version2
    int oldVersion {Program::Version1::getVersion()}; // Uses getVersion() from Version1
    int newVersion {Program::Version2::getVersion()}; // Uses getVersion() from Version1
    //bool firstVersion {Program::isFirstVersion()};    // Does not compile when Version2 is added  !!!ERROR!!!\

    printf("version    = %d\n", version);
    printf("oldVersion = %d\n", oldVersion);
    printf("newVersion = %d\n", newVersion);
  }

  {
    bool firstVersion {Program::Version1::isFirstVersion()};\
  }

  return 0;
}
