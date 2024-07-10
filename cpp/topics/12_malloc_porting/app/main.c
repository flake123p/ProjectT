
#define USING_STL_VECTOR ( 1 )

#include "demo_cpp.hpp"

#include <stdint.h>
#include <vector>

/* Claim Register - 0x1000 per target */
#define PLIC_CLAIM_OFFSET               (0x00200004UL)
#define PLIC_CLAIM_SHIFT_PER_TARGET     12

#define _IO_(addr)              (addr)
#define PLIC_SW_BASE            _IO_(0xE6400000)
#define PLIC_CLAIM_OFFSET               (0x00200004UL)

#define NDS_MHARTID 0xF14


#define read_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })
#define rd_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })

#define  NDS_MIE              0x304
// #define IRQ_M_SOFT          	3
// #define MIP_MSIP            	(1 << IRQ_M_SOFT)

#define set_csr(reg, bit) ({ unsigned long __tmp; \
  asm volatile ("csrrs %0, " #reg ", %1" : "=r"(__tmp) : "rK"(bit)); \
  __tmp; })

#define clear_csr(reg, bit) ({ unsigned long __tmp; \
  asm volatile ("csrrc %0, " #reg ", %1" : "=r"(__tmp) : "rK"(bit)); \
  __tmp; })
  
#define HAL_MSWI_ENABLE()                               set_csr(0x304, (1 << 3))


#define IRQ_S_SOFT   1
#define IRQ_H_SOFT   2
#define IRQ_M_SOFT   3
#define IRQ_S_TIMER  5
#define IRQ_H_TIMER  6
#define IRQ_M_TIMER  7
#define IRQ_S_EXT    9
#define IRQ_H_EXT    10
#define IRQ_M_EXT    11
#define IRQ_COP      12
#define IRQ_HOST     13
#define MIP_SSIP            (1U << IRQ_S_SOFT)
#define MIP_HSIP            (1U << IRQ_H_SOFT)
#define MIP_MSIP            (1U << IRQ_M_SOFT)
#define MIP_STIP            (1U << IRQ_S_TIMER)
#define MIP_HTIP            (1U << IRQ_H_TIMER)
#define MIP_MTIP            (1U << IRQ_M_TIMER)
#define MIP_SEIP            (1U << IRQ_S_EXT)
#define MIP_HEIP            (1U << IRQ_H_EXT)
#define MIP_MEIP            (1U << IRQ_M_EXT)

#define CLINT_BASE         0xF0010000
typedef struct CLINT_Type_t
{
    volatile uint32_t MSIP[5];
    volatile uint32_t reserved1[(0x4000U - 0x14U)/4U];
    volatile uint64_t MTIMECMP[5];  /* mtime compare value for each hart. When mtime equals this value, interrupt is generated for particular hart */
    volatile uint32_t reserved2[((0xbff8U - 0x4028U)/4U)];
    volatile uint64_t MTIME;    /* contains the current mtime value */
} CLINT_Type;
#define CLINT    ((CLINT_Type *)CLINT_BASE)
static inline void clear_soft_interrupt(void)
{
    volatile uint32_t reg;
    uint64_t hart_id = read_csr(mhartid);
    CLINT->MSIP[hart_id] = 0x00U;   /*clear soft interrupt for hart0*/
    reg = CLINT->MSIP[hart_id];     /* we read back to make sure it has been written before moving on */
                                    /* todo: verify line above guaranteed and best way to achieve result */
    (void)reg;                      /* use reg to avoid compiler warning */
}
#define MSTATUS_MIE         0x00000008
void __enable_irq(void)
{
    set_csr(mstatus, MSTATUS_MIE);  /* mstatus Register- Machine Interrupt Enable */
}


void cpu1_cmd_handler()
{
    //extern void embcv_cpu_cmd_handler();
    //embcv_cpu_cmd_handler();
}

volatile int enable_cpu1 = 0;
volatile int abxx = 0;
volatile int abcc = 0;
volatile unsigned int hid;
//__attribute__((always_inline)) static inline unsigned int __nds__plic_sw_claim_interrupt(void)
extern "C" __attribute__((used)) void __nds__plic_sw_claim_interrupt(void)
{
//   //HAL_MSWI_ENABLE();
//   set_csr(0x344, (1 << 3));
//   unsigned int hart_id = read_csr(0xF14);
//   volatile unsigned int *claim_addr = (volatile unsigned int *)(PLIC_SW_BASE +
//                                                                 PLIC_CLAIM_OFFSET +
//                                                                 (hart_id << PLIC_CLAIM_SHIFT_PER_TARGET));
//   return  *claim_addr;

    clear_soft_interrupt();
    set_csr(mie, MIP_MSIP);

    hid = read_csr(mhartid);
    switch (hid) {
        case 0:
            abxx += 1;
            break;
        case 1:
            abxx += 10;
            {
                cpu1_cmd_handler();
                while(1){;}
            }
            break;
        case 2:
            abxx += 100;
            break;
        case 3:
            abxx += 1000;
            break;
    }

    /* Put this hart in WFI */
#if 1
    __asm("wfi");
    // clear_soft_interrupt();
    // __enable_irq();
#else
    do
    {
        __asm("wfi");
    }while(0 == (read_csr(mip) & MIP_MSIP));

    /* The hart is out of WFI, clear the SW interrupt. Here onwards application
     * can enable and use any interrupts as required */

    clear_soft_interrupt();

    __enable_irq();
#endif
    //return 0;
}

extern "C" void __startX() __attribute__((naked, used));
extern "C" void __startX()
{
	// asm volatile("la t0, _stack-16");
	// asm volatile("mv sp, t0");
	// asm volatile ("auipc sp, 0x4010");
	asm volatile("j __startXX");
}

volatile int aa = 0;
volatile int bb = 0;
volatile int cc = 0;
volatile int ddd = 0;

#if USING_STL_VECTOR
int foobar(std::vector<int> &in)
{
    int ret = 0;
    for(std::size_t i = 0; i < in.size(); ++i) {
        ret += in[i];
    }
    return ret;
}
#else
typedef struct {
    LList_Entry_t en;
    int val;
} test_t;
int foobar(LList_Head_t *hd)
{
    int ret = 0;
    test_t *curr;
    LLIST_FOREACH(hd, curr, test_t) {
        ret += curr->val;
    }
    return ret;
}
#endif

int main()
{
    void *p = malloc(10);
    p = p;
    free(p);
    unsigned int hid = read_csr(mhartid);
    if (hid == 1) {
        //while(1){__asm("wfi");}
        while(1){;}
    }

    aa = rd_csr(0xB00);
    bb = rd_csr(0xB00);

#if USING_STL_VECTOR
    std::vector<int> myv;
    myv.push_back(1);
    myv.push_back(2);
    myv.push_back(3);
    aa = foobar(myv);
#else
    LList_Head_t hd = LLIST_HEAD_INIT(&hd);
#   if 1
    test_t a;
    test_t b;
    test_t c;
    a.val = 1;
    b.val = 2;
    c.val = 3;
    DLLIST_INSERT_LAST(&hd, &a);
    DLLIST_INSERT_LAST(&hd, &b);
    DLLIST_INSERT_LAST(&hd, &c);
#   else
    test_t *a = (test_t *)MM_ALLOC(sizeof(test_t));
    test_t *b = (test_t *)MM_ALLOC(sizeof(test_t));
    test_t *c = (test_t *)MM_ALLOC(sizeof(test_t));
    a->val = 1;
    b->val = 2;
    c->val = 4;
    DLLIST_INSERT_LAST(&hd, a);
    DLLIST_INSERT_LAST(&hd, b);
    DLLIST_INSERT_LAST(&hd, c);
#   endif
    aa = foobar(&hd);
#endif
    cc = rd_csr(0xB00);
    ddd = cc - bb;
    //extern void __startXXX();
    //__startXXX();

    //clear_soft_interrupt();
    //set_csr(mie, MIP_MSIP);

    limits_demo(0);
    float_demo(0);
    complex_demo(0);
    type_traits_demo(3);

    atomic_demo(0);
    functional_demo(0);
    utility_demo(2);
    algorithm_demo(0);
    iterator_demo(0);

    set_demo(1);
    initializer_list_demo(0);
    deque_demo(0);
    map_demo(0);
    unordered_map_demo(0);
    
    string_demo(0);
    sstream_demo(0);

    //CLINT->MSIP[1] = 1;
    while(1){;}
	return 0;
}

