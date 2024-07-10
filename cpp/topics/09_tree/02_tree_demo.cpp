#include "_list.h"
#include "_tree.h"
#include "_treeDraw.h"

int main()
{
    {
        class TreeClass a;
        BASIC_ASSERT(a.calc_max_depth() == 0);
    }
    {
        class TreeClass a, b, c;
        a.set_left(b);
        a.set_right(c);
        BASIC_ASSERT(a.calc_max_depth() == 1);
    }
    {
        class TreeClass a, b, c, d;
        a.set_left(b);
        b.set_right(c);
        b.set_left(d);
        BASIC_ASSERT(a.calc_max_depth() == 2);
        BASIC_ASSERT(b.calc_max_depth() == 1);
        BASIC_ASSERT(c.calc_max_depth() == 0);

        class TreeDraw<class TreeClass *> treeDraw(a.calc_max_depth());
        treeDraw.DumpWithIndice();
        treeDraw.TreeNodesRegister(
            &a,
            [](class TreeClass *prev, int isLeft) -> class TreeClass * {
                if (isLeft) {
                    return prev->left;
                } else {
                    return prev->right;
                }
            }
        );
        treeDraw.Draw();

        treeDraw.DrawV(
            [](class TreeClass *node) -> void {
                node = node;
            }
        );
    }
}