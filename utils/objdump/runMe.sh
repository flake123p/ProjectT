#!/bin/bash

#
# REF: https://ithelp.ithome.com.tw/articles/10196279
#
g++ -std=c++17 demo.cpp && ./a.out

# Z: source code with disassembly 
# l: line-numbers
# z: disassemble-zeroes
objdump -Slz a.out > 0_Slz.log

objdump -SlzafphxgeGWtTrRs a.out > 1_ALL.log

# $ unset GTK_PATH && meld *.log  // Observe ...

objdump -Slza a.out > 10_a.log # -a, --archive-headers    Display archive header information 
objdump -Slzf a.out > 11_f.log # -f, --file-headers       Display the contents of the overall file header 
objdump -Slzp a.out > 12_p.log # -p, --private-headers    Display object format specific file header contents 
objdump -Slzh a.out > 13_h.log # -h, --[section-]headers  Display the contents of the section headers 
objdump -Slzx a.out > 14_x.log # -x, --all-headers        Display the contents of all headers 
objdump -Slzg a.out > 15_g.log # -g, --debugging          Display debug information in object file 
objdump -Slze a.out > 16_e.log # -e, --debugging-tags     Display debug information using ctags style !!!!!!!!!!!!1
objdump -SlzG a.out > 17_G.log # -G, --stabs              Display (in raw form) any STABS info in the file 
objdump -SlzW a.out > 18_W.log # -W, --dwarf              Display DWARF info in the file 
objdump -Slzt a.out > 19_t.log # -t, --syms               Display the contents of the symbol table(s) 
objdump -Slzr a.out > 20_r.log # -r, --reloc              Display the relocation entries in the file 
objdump -SlzR a.out > 21_R.log # -R, --dynamic-reloc      Display the dynamic relocation entries in the file 
objdump -Slzs a.out > 22_s.log # -s, --full-contents      Display the full contents of all sections requested 

objdump -Slzd a.out > 50_d.log # -d, --disassemble        Display assembler contents of executable sections 
objdump -SlzD a.out > 51_D.log # -D, --disassemble-all    Display assembler contents of all sections
objdump -SlzD --prefix-addresses a.out > 51a_D.log
objdump -SlzD --disassemble-zeroes  a.out > 51b_D.log
objdump -SlzD --prefix-addresses --disassemble-zeroes a.out > 51c_D.log