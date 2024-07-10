#
# https://stackoverflow.com/questions/137038/how-do-you-get-assembler-output-from-c-c-source-in-gcc
#

gcc -S -O2 -g 01_pow_i_lambda.cpp -I../../projects/p01/mod/basic

gcc -c -O2 -g 01_pow_i_lambda.cpp -I../../projects/p01/mod/basic

objdump -S -drwC 01_pow_i_lambda.o>01_pow_i_lambda.dump