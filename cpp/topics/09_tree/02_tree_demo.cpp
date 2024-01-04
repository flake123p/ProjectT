#include "_list.h"
#include "_tree.h"

int main()
{
    {
        class TreeClass a, b, c;
        a.set_left(b);
        a.set_right(c);
        BASIC_ASSERT(a.calc_max_depth() == 2);
    }
    {
        class TreeClass a, b, c;
        a.set_left(b);
        b.set_right(c);
        BASIC_ASSERT(a.calc_max_depth() == 3);
        BASIC_ASSERT(b.calc_max_depth() == 2);
        BASIC_ASSERT(c.calc_max_depth() == 1);
    }
}