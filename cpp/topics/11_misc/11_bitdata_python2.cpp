#include <cstring>
#include <iostream>       // std::cout

#define VEC_ADD (6)

#define MEM_SCATTER (3)
#define MEM_GATHER  (4)

typedef struct __attribute__((packed)) {
    uint8_t core_busy[2]; // 1 bit x 10 cores
    uint8_t core_op[10];
    uint8_t mem_busy; // 1 bit
    uint8_t mem_dst; // 8 bits
    uint8_t mem_src; // 8 bits
    uint8_t mem_op;  // 8 bits, 3 for Scatter, 4 for Gather
} TimingData ;

#define CORE_0 0
#define CORE_1 1
#define CORE_2 2
#define CORE_3 3
#define CORE_4 4
#define CORE_5 5
#define CORE_6 6
#define CORE_7 7
#define CORE_8 8
#define CORE_9 9

#define CORE_0_IDX 0
#define CORE_1_IDX 0
#define CORE_2_IDX 0
#define CORE_3_IDX 0
#define CORE_4_IDX 0
#define CORE_5_IDX 0
#define CORE_6_IDX 0
#define CORE_7_IDX 0
#define CORE_8_IDX 1
#define CORE_9_IDX 1

#define CORE_0_SHIFT 0
#define CORE_1_SHIFT 1
#define CORE_2_SHIFT 2
#define CORE_3_SHIFT 3
#define CORE_4_SHIFT 4
#define CORE_5_SHIFT 5
#define CORE_6_SHIFT 6
#define CORE_7_SHIFT 7
#define CORE_8_SHIFT 0
#define CORE_9_SHIFT 1

int main() 
{
    TimingData dat;

    memset(&dat, 0, sizeof(TimingData));
    dat.mem_src = 0xFF;
    dat.mem_dst = 0xFF;

    FILE *fp = fopen("timing2_00.bin", "wb");    // hexdump timing00.bin -v -e '/1 "%02x\n"'
    fwrite(&dat, 1, sizeof(TimingData), fp);
    fclose(fp);

    //
    // 1
    //
    dat.mem_src = CORE_0;
    dat.mem_dst = CORE_5;
    dat.mem_busy = 1;
    dat.mem_op = MEM_SCATTER;

    fp = fopen("timing2_01.bin", "wb");
    fwrite(&dat, 1, sizeof(TimingData), fp);
    fclose(fp);

    //
    // 2
    //
    memset(&dat, 0, sizeof(TimingData));
    dat.mem_src = 0xFF;
    dat.mem_dst = 0xFF;

    uint8_t op_busy = 1 << CORE_5_SHIFT;
    dat.core_busy[CORE_5_IDX] |= op_busy;
    dat.core_op[CORE_5] = VEC_ADD;

    dat.mem_src = CORE_0;
    dat.mem_dst = CORE_3;
    dat.mem_busy = 1;
    dat.mem_op = MEM_SCATTER;

    fp = fopen("timing2_02.bin", "wb");
    fwrite(&dat, 1, sizeof(TimingData), fp);
    fclose(fp);


    //
    // 3
    //
    memset(&dat, 0, sizeof(TimingData));
    dat.mem_src = CORE_5;
    dat.mem_dst = CORE_0;
    dat.mem_busy = 1;
    dat.mem_op = MEM_GATHER;

    op_busy = 1 << CORE_3_SHIFT;
    dat.core_busy[CORE_3_IDX] |= op_busy;
    dat.core_op[CORE_3] = VEC_ADD;

    fp = fopen("timing2_03.bin", "wb");
    fwrite(&dat, 1, sizeof(TimingData), fp);
    fclose(fp);

    //
    // 4
    //
    memset(&dat, 0, sizeof(TimingData));
    dat.mem_src = CORE_3;
    dat.mem_dst = CORE_0;
    dat.mem_busy = 1;
    dat.mem_op = MEM_GATHER;
    fp = fopen("timing2_04.bin", "wb");
    fwrite(&dat, 1, sizeof(TimingData), fp);
    fclose(fp);

    //
    // 5
    //
    memset(&dat, 0, sizeof(TimingData));
    dat.mem_src = 0xFF;
    dat.mem_dst = 0xFF;

    fp = fopen("timing2_05.bin", "wb");
    fwrite(&dat, 1, sizeof(TimingData), fp);
    fclose(fp);

    return 0;
}