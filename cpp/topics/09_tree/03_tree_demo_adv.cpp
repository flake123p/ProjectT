#include "_list.h"
#include "_tree.h"
#include "_treeDraw.h"

int main()
{
    {
        class TreeClass a, b, c, d, e, f, g;
        int aa = 111;
        a.payload = &aa;
        int bb = 222;
        b.payload = &bb;
        int cc = 333;
        c.payload = &cc;
        int dd = 444;
        d.payload = &dd;
        int ee = 555;
        e.payload = &ee;
        int ff = 666;
        f.payload = &ff;
        int gg = 777;
        g.payload = &gg;
        a.set_left(b);
        b.set_right(c);
        b.set_left(d);

        a.set_right(e);
        e.set_right(f);
        e.set_left(g);

        class TreeDraw<class TreeClass *> treeDraw(a.calc_max_depth());

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

        treeDraw.DrawV(
            [](class TreeClass *node) -> void {
                node = node;
                if (node == nullptr) {
                    printf("     ");
                } else {
                    int *payload = (int *)node->payload;
                    printf("<%3d>", *payload);
                }
            }
        );

        printf("Traversal Dump:\n");
        a.traverse (
            [](class TreeClass *cur, int cur_depth) -> void {
                cur_depth = cur_depth; //dummy

                int *payload = (int *)cur->payload;
                printf("<< %3d >>\n", *payload);
            }
        );
    }
}