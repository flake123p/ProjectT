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
#define PRLOC printf("%s() %d\n", __func__, __LINE__);



void lgl::LogicGatesLearn_Demo()
{
    PRINT_FUNC

    class LearnGrid gr(3, 4);

    LearnGrid::random_init();

    lgs::LogicUnitBase inputA(lgs::input, 0); inputA.name = "_A";
    lgs::LogicUnitBase inputB(lgs::input, 1); inputB.name = "_B";
    gr.emplace_input_list(&inputA);
    gr.emplace_input_list(&inputB);

    gr.randomly_initialize_types();
    gr.randomly_initialize_connections();
    gr.unset_all_used_unit();
    gr.traverse_used_unit(gr.get_unit(0, 3));
    gr.dump();

    gr.initialize_lgs_grid();

    lgs::LogicUnitBase *out = gr.get_lgs_unit(0, 3);
    out->type_dump();
    COUT(out->evaluate());
}

void lgl::LogicGatesLearn_DemoBruteForce()
{
    PRINT_FUNC

    class lgl::LearnGrid gr(3, 4);

    lgl::LearnGrid::random_init();

    lgs::LogicUnitBase inputA(lgs::input, 0); inputA.name = "_A";
    lgs::LogicUnitBase inputB(lgs::input, 1); inputB.name = "_B";
    gr.emplace_input_list(&inputA);
    gr.emplace_input_list(&inputB);

    int it = 0;
    lgs::LogicUnitBase *out;
    int match_ctr;
    while (true) {
        COUT(it++);
        gr.randomly_initialize_types();
        gr.randomly_initialize_connections();
        gr.initialize_lgs_grid();
        //
        // input output evaluate
        //
        out = gr.get_lgs_unit(0, 3);
        match_ctr = 0;
        // 0 0 - 1
        // 0 1 - 0   Truth Table
        // 1 0 - 1
        // 1 1 - 1
        inputA.value = 0; inputB.value = 0; if (out->evaluate() == 1){match_ctr++;}
        inputA.value = 0; inputB.value = 1; if (out->evaluate() == 0){match_ctr++;}
        inputA.value = 1; inputB.value = 0; if (out->evaluate() == 1){match_ctr++;}
        inputA.value = 1; inputB.value = 1; if (out->evaluate() == 1){match_ctr++;}

        if (match_ctr == 4) {
            printf("finish ...\n");
            gr.dump();
            out->type_dump();
            break;
        }
    }
}