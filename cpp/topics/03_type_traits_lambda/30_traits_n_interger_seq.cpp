
#include <iostream>
#include <ostream>
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
#include <random>
#include <iterator>
#include <charconv>
#include <cassert>

#include "FunctionTraits.h"


/*
https://www.cnblogs.com/happenlee/p/14219925.html

std::integer_sequence
std::make_integer_sequence
std::make_index_sequence
std::index_sequence_for
*/
template <size_t ...N>
static constexpr auto square_nums(size_t index, std::index_sequence<N...>) {
    constexpr auto nums = std::array{N * N ...};
    return nums[index];
}

template <size_t N>
constexpr static auto const_nums(size_t index) {
    return square_nums(index, std::make_index_sequence<N>{});
}

int main() {
    static_assert(const_nums<101>(100) == 100 * 100);

    auto x = const_nums<3>;
    std::cout << x(0) << std::endl;
    std::cout << x(1) << std::endl;
    std::cout << x(2) << std::endl;

}