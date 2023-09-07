
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