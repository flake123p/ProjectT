
# Run with readelf
# $ readelf -Ws a.out | c++filt >out.txt

c++filt _Z4abxxv

#
# PLEASE REMOVE @GLIBCXX_3.4 (3)
# PLEASE REMOVE @GLIBCXX_3.4 (3)
# PLEASE REMOVE @...... (WHATEVER)
#
c++filt '_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GLIBCXX_3.4 (3)'
c++filt _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_

echo ...
c++filt _ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_21compare_scalar_kernelIdEEvRNS_18TensorIteratorBaseENS0_50_GLOBAL__N__e50ef81d_17_CompareKernels_cu_8f1b29aa6OpTypeET_EUldE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_