
# Flake

## CUDA examples
[using/function/lambda/ref/const/template]
https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtIgemmTensor
```
#include <functional>

using SampleRunner = std::function<void()>;

void run(const SampleRunner& runSample) {
    ...
}

int main() {
    TestBench<int8_t, int32_t> props(4, 4, 4);

    props.run([&props] {
    LtIgemmTensor(props.ltHandle,
}
```

## MOVE/RVO/NRVO
https://learn.microsoft.com/en-us/previous-versions/ms364057(v=vs.80)?redirectedfrom=MSDN
Move semantics also helps when the compiler cannot use Return Value Optimization (RVO) or Named Return Value Optimization (NRVO). 
In these cases, the compiler calls the move constructor if the type defines it. 
For more information about Named Return Value Optimization, see Named Return Value Optimization in Visual C++ 2005.