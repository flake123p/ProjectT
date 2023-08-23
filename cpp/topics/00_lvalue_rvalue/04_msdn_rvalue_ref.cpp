
#include "all.hpp"

//
// https://learn.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2010/dd293668(v=vs.100)
//

/*
    Holds a reference to an rvalue expression.

    type-id && cast-expression

    The following sections describe how rvalue references support the implementation of [[move semantics]] and [[perfect forwarding]].

    [[Move semantics]] works because it enables resources to be transferred from temporary objects that cannot be referenced elsewhere in the program.
*/

using namespace std;

int main()
{
    /*
    By using rvalue references, operator+ can be modified to take rvalues, which cannot be referenced elsewhere in the program. 
    Therefore, operator+ can now append one string to another.
    */
    string s = string("h") + "e" + "ll" + "o";
    cout << s << endl;
    return 0;
}
