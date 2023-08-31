
# From cuda samples:

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