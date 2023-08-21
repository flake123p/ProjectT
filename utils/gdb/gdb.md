
# GDB命令基礎，讓你的程序bug無處躲藏
https://www.cntofu.com/book/46/gdb/gdbming_ling_ji_chu_ff0c_rang_ni_de_cheng_xu_bug_w.md

# GDB實用教學：自動化你的debug
https://jasonblog.github.io/note/gdb/gdbshi_yong_jiao_xue_ff1a_zi_dong_hua_ni_de_debug.html


# sample 1 (gdbrc)
```
file a
target remote localhost:5555
tui enable
layout split
#set history filename gdbhist
#set history save on
#set history remove-duplicates 7
#break mtvec_interrupt_handler
#break __nds__plic_sw_claim_interrupt
#break prof_func_3
# break __nds__plic_sw_claim_interrupt
# #break prof_func_3
# break main.c:102
# break __register_exitproc
# break main.c:197
#break atexit
break main
# break main.c:242
# break test_d2d_prof.c:28
# break test_d2d_prof.c:78
# break detect_2d.hpp:84
info break
```

# sample 2 (gdbrc)
```
file a
target remote localhost:5555
#tui enable
#layout split

set logging overwrite on
set logging enabled
set confirm off

define fn
    p "xx, yy"
    p {xx, yy}
    p "mem, t"
    p {maxmem, ddd}
    p sys_rltv_cycle
    p sys_max_cycle_idx
    p the_roi
    end

fn

q
```

# Give input to gdb debugger script
https://stackoverflow.com/questions/67794782/give-input-to-gdb-debugger-script
(gdb) run < testinput.txt