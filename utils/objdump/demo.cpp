
#include <iostream>
#include <cstdint>
#include <memory>

int myadd(int a, int b)
{
    a = a * 10;
    b = b * 10;
    return a + b;
}

void abxx()
{
    printf("abxx\n");
}

int main() {
    abxx();

    printf("myadd = %d\n", myadd(3, 4));
    return 0;
}