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
// #include <execution>
// #include <tbb/tbb.h>
// using namespace tbb::v1;
#include <random>
#include <iterator>
#include <charconv>
#include <stdio.h>
#include <stdint.h>

#include "LogicGatesSim.hpp"
#include "LogicGatesLearn.hpp"

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PR(a)   std::cout << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

void demo()
{
    lgs::LogicUnitBase a(lgs::input, 0); 
    a.name = "_a";
    lgs::LogicUnitBase b(lgs::input, 1); 
    b.name = "_b";

    lgs::LogicUnitBase notG(lgs::notGate);
    notG.import(a);
    COUT(notG.evaluate());

    lgs::LogicUnitBase andG(lgs::andGate);
    andG.import(a, b);
    COUT(andG.evaluate());

    lgs::LogicUnitBase orG(lgs::orGate);
    orG.import(a, b);
    COUT(orG.evaluate());

    lgs::LogicUnitBase xorG(lgs::xorGate);
    xorG.import(a, b);
    COUT(xorG.evaluate());

    lgs::LogicUnitBase orG2(lgs::orGate);
    orG2.import(andG, b);
    COUT(orG2.evaluate());

    COUT(orG2.type2str().c_str());

    orG2.type_dump();
}

int main(int argc, char* argv[])
{
    demo();
    lgl::LogicGatesLearn_DemoBruteForce();

    return 0;
}
