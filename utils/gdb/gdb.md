
# GDB命令基礎，讓你的程序bug無處躲藏
https://www.cntofu.com/book/46/gdb/gdbming_ling_ji_chu_ff0c_rang_ni_de_cheng_xu_bug_w.md

(gdb) r/run             # 開始運行程序
(gdb) c/continue        # 繼續運行
(gdb) n/next            # 下一行，不進入函數調用
(gdb) s/step            # 下一行，進入函數調用
(gdb) ni/si             # 嚇一跳指令，ni和si區別同上
(gdb) fini/finish       # 繼續運行至函數退出/當前棧幀
(gdb) u/util            # 繼續運行至某一行，在循環中，u可以實現運行至循環剛剛退出，但這取決於循環的實現

(gdb) set args          # 設置程序啟動參數，如：set args 10 20 30
(gdb) show args         # 查看程序啟動參數
(gdb) path <dir>        # 設置程序的運行路徑
(gdb) show paths        # 查看程序的運行路徑
(gdb) set env <name=val># 設置環境變量，如：set env USER=chen
(gdb) show env [name]   # 查看環境變量
(gdb) cd <dir>          # 相當於shell的cd命令
(gdb) pwd               # 顯示當前所在目錄

(gdb) shell <commond>   # 執行shell命令

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