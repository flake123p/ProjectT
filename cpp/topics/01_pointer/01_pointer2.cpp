//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

uint8_t gMem[] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,   0x08, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,   0x28, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3f,
};

int main() 
{
    // pointer to u8
    uint8_t *pgMem8 = (uint8_t *)gMem;
    printf("U8  pointer : 0x%02X, 0x%02X, 0x%02X, 0x%02X\n", pgMem8[0], pgMem8[1], pgMem8[2], pgMem8[3]);
    printf("\n");

    // pointer to u16 (result is by the CPU endian)
    uint16_t *pgMem16 = (uint16_t *)gMem;
    printf("U16 pointer : 0x%04X, 0x%04X, 0x%04X, 0x%04X\n", pgMem16[0], pgMem16[1], pgMem16[2], pgMem16[3]);

    if (pgMem16[0] == 0x0201) {
        printf("This machine is Little Endian\n");
    } else {
        printf("This machine is Big Endian\n");
    }
    printf("\n");

    //
    uint8_t *p1 = (uint8_t *)&gMem[0];
    uint8_t *p2 = (uint8_t *)&gMem[8];
    uint8_t *p3 = (uint8_t *)&gMem[16];
    uint8_t *p4 = (uint8_t *)&gMem[24];
    uint8_t *pArray[] = {p1, p2, p3, p4};

    uint8_t **ppgMem8 = &p1;
    printf("Pointer & Array demo 1:\n");
    printf("ppgMem8[0][0] : 0x%02X\n", ppgMem8[0][0]);
    printf("ppgMem8[0][1] : 0x%02X\n", ppgMem8[0][1]);
    //printf("*ppgMem8[2]   : 0x%02X\n", *ppgMem8[2]); //invalid
    printf("(*ppgMem8)[2] : 0x%02X\n", (*ppgMem8)[2]);
    printf("**ppgMem8     : 0x%02X\n", **ppgMem8);

    //
    // C++ Operator Precedence https://en.cppreference.com/w/cpp/language/operator_precedence
    //
    // https://en.cppreference.com/w/cpp/language/operator_member_access#Built-in_subscript_operator
    //
    ppgMem8 = pArray;
    printf("Pointer & Array demo 2:\n");
    printf("ppgMem8[0][0] : 0x%02X\n", ppgMem8[0][0]);
    printf("ppgMem8[0][1] : 0x%02X\n", ppgMem8[0][1]);
    printf("*ppgMem8[2]   : 0x%02X (subscript first)\n", *ppgMem8[2]);  //subscript first
    printf("(*ppgMem8)[2] : 0x%02X\n", (*ppgMem8)[2]);
    printf("*(ppgMem8[2]) : 0x%02X\n", *(ppgMem8[2]));
    printf("ppgMem8[0][0] : 0x%02X\n", ppgMem8[0][0]);
    printf("ppgMem8[1][0] : 0x%02X\n", ppgMem8[1][0]);
    printf("ppgMem8[2][0] : 0x%02X\n", ppgMem8[2][0]);
    printf("ppgMem8[3][0] : 0x%02X\n", ppgMem8[3][0]);
    printf("**ppgMem8     : 0x%02X\n", **ppgMem8);



    return 0;
}