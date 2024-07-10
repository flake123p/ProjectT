
#define STACKSIZE        0x100000

.equ STACKSIZE, 0x1000000

.equ UART_BASE, 0x60001800
.equ RxTx, 0x0
.equ TxFull, 0x04
.equ RxEmpty, 0x08
.equ EventStatus, 0x0c
.equ EventPending, 0x10
.equ EventEnable, 0x14

##define MSTATUS_FS          	0x00006000
.equ MSTATUS_FS, 0x00006000

#.equ memtop, 0x00040000
.equ memtop, 0x3000FFF0
.equ XLEN, 4 # 32 bit, 4*8

.section .text
#.global _start      # Provide program starting address to linker
.global __startXX
__startXX:

        # setup machine trap vector
1:      auipc   t0, %pcrel_hi(mtvec_interrupt_handler)  # load mtvec_interrupt_handler(hi)
        addi    t0, t0, %pcrel_lo(1b)                   # load mtvec_interrupt_handler(lo)
        csrw   mtvec, t0

        # set mstatus.MIE=1 (enable M mode interrupts in general)
        li      t0, 0b0000000000001000
        csrrs   zero, mstatus, t0

        # set mie.MTIE=1 (enable M mode timer interrupts)
        li      t0, 0b0000000010000000
        csrrs   zero, mie, t0

        # set mie.MEIE=1 (enable M mode external interrupts)
        li      t0, 0b0000100000000000
        csrrs   zero, mie, t0

        # set mie.MSIE=1 (enable M mode machine software interrupt)
        li      t0, 0b0000000000001000
        csrrs   zero, mie, t0

        # setup a stack pointer
        #la sp, memtop

	/* Initialize stack pointer */
	la t0, memtop
	mv sp, t0
	/* get cpu id */
	csrr t0, mhartid
	/* sp = _stack - (cpuid * STACKSIZE) */
	li t1, STACKSIZE
	mul t0, t0, t1
	sub sp, sp, t0

	/* Enable FPU */
	li t0, MSTATUS_FS
	csrrs t0, mstatus, t0

	/* Initialize FCSR */
	fscsr zero

        # no process is running by default
        # squat on tp to hold which process is running
        # linux kinda does this sooo..
        li tp, 0

        # setup gp
        .option push
        .option norelax
        #la gp, __global_pointer$
        .option pop

        #call init_uart
        #call init_processes
        j _start
        #j main
forever:
        j forever

mtvec_interrupt_handler:
        call __nds__plic_sw_claim_interrupt  
        
        #skip saving if not process started yet
        # Flake: change these in 01_SMPx
        # call __nds__plic_sw_claim_interrupt  
        # beq     tp, zero, no_current_process

        # store pc from the process onto the stack
        csrr    t0, mepc
        sw      t0, 29*XLEN(sp)
        # store sp from the process
        # current_process->sp = sp
        sw      sp, 0(tp)

        call __nds__plic_sw_claim_interrupt  

        # restore sp from the process
        # sp = current_process->sp
        lw      sp, 0(tp)
        # restore pc into mepc
        lw      t0, 29*XLEN(sp)
	csrw    mepc, t0
        mret
        
.global __startXXX
__startXXX:
        # setup machine trap vector
        auipc   t0, %pcrel_hi(mtvec_interrupt_handler)  # load mtvec_interrupt_handler(hi)
        addi    t0, t0, %pcrel_lo(1b)                   # load mtvec_interrupt_handler(lo)
        csrw   mtvec, t0

        # set mstatus.MIE=1 (enable M mode interrupts in general)
        li      t0, 0b0000000000001000
        csrrs   zero, mstatus, t0

        # set mie.MTIE=1 (enable M mode timer interrupts)
        li      t0, 0b0000000010000000
        csrrs   zero, mie, t0

        # set mie.MEIE=1 (enable M mode external interrupts)
        li      t0, 0b0000100000000000
        csrrs   zero, mie, t0

        # set mie.MSIE=1 (enable M mode machine software interrupt)
        li      t0, 0b0000000000001000
        csrrs   zero, mie, t0

        ret
