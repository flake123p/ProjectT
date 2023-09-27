
# 0927
- [ ] CTAD, std::lock_guard, std::mutex 
https://tjsw.medium.com/%E6%BD%AE-c-17-class-template-argument-deduction-%E5%92%8C-deduction-guide-%E9%A1%9E%E5%88%A5%E6%A8%A3%E7%89%88%E5%8F%83%E6%95%B8%E6%8E%A8%E5%B0%8E-70cc36307a42

# From pytorch:

```
  static size_t defaultNumThreads() {
    auto num_threads = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    num_threads /= 2;
#endif
    return num_threads;
  }
};

class C10_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;
```