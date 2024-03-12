# file image
# target remote localhost:5555
# tui enable
# layout split
# #set history filename gdbhist
# #set history save on
# #set history remove-duplicates 7
# #break mtvec_interrupt_handler
# #break __nds__plic_sw_claim_interrupt
# info break

file a.out
tui enable
layout split
start
l