//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

class VirMachine 
{
private:
    const int MEM_MAX = 256;
    uint8_t *mem;
public:
    VirMachine() {
        mem = (uint8_t *)calloc(MEM_MAX, sizeof(uint8_t));
        printf("Hello Virtual Machine - Constructor\n");
    }
    ~VirMachine() {
        free(mem);
        printf("Hello Virtual Machine - Destructor\n");
    }

    uint8_t *Addr(int index) {
        if (index >= MEM_MAX) {
            return NULL;
        }
        return mem+index;
    }

    void Dump() {
        printf("              0  1  2  3  4  5  6  7   8  9  A  B  C  D  E  F\n");
        printf("             ------------------------------------------------\n");
        printf("0x00000000 | ");
        for (int i = 0; i < MEM_MAX; i++) {
            printf("%02X ", mem[i]);
            if (i % 16 == 7) {
                printf(" ");
            } else if (i % 16 == 15) {
                printf("\n");
                if (i != MEM_MAX -1) {
                    printf("0x%08X | ", (i / 16 + 1) * 0x10);
                }
            }
        }
    }
};

int main() 
{
    VirMachine mac;

    uint32_t *p32 = (uint32_t *)mac.Addr(4);
    *p32 = 0x11223344;

    mac.Dump();

    return 0;
}
