
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <climits>
#include <cfloat>
#include <variant>
#include <optional>
#include <any>
#include <filesystem>
#include <cstddef>
#include <set>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  Splicing for maps and sets

  Moving nodes and merging containers without the overhead of expensive copies, moves, or heap allocations/deallocations.

  Moving elements from one map to another:
*/

int main() 
{
  using namespace std;

  std::map<int, string> src {{1, "one"}, {2, "two"}, {3, "buckle my shoe"}};
  std::map<int, string> dst {{3, "three"}};
  dst.insert(src.extract(src.find(1))); // Cheap remove and insert of { 1, "one" } from `src` to `dst`.
  dst.insert(src.extract(2)); // Cheap remove and insert of { 2, "two" } from `src` to `dst`.
  // dst == { { 1, "one" }, { 2, "two" }, { 3, "three" } };

  for (auto& _src : src) {
    COUT(_src.first);
    COUT(_src.second);
  }
  for (auto& _dst : dst) {
    COUT(_dst.first);
    COUT(_dst.second);
  }

  //
  // Inserting an entire set:
  //
  {
    std::set<int> src {1, 3, 5};
    std::set<int> dst {2, 4, 5};
    dst.merge(src);
    // src == { 5 }
    // dst == { 1, 2, 3, 4, 5 }
    for (auto& __dst : dst) {
      COUT(__dst);
    }
  }

  //
  // Inserting elements which outlive the container:
  //
  {/*
    auto elementFactory() {
      std::set<...> s;
      s.emplace(...);
      return s.extract(s.begin());
    }
    s2.insert(elementFactory());
  */}

  //
  // Changing the key of a map element:
  //
  {
    std::map<int, string> m {{1, "one"}, {2, "two"}, {3, "three"}};
    auto e = m.extract(2);
    e.key() = 4;
    m.insert(std::move(e));
    // m == { { 1, "one" }, { 3, "three" }, { 4, "two" } }
  }

  return 0;
}
