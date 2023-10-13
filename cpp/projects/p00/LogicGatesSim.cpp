
#include "LogicGatesSim.hpp"

template <typename T>
class LogicGatesSim {
public:
    T ary[10][10];
    LogicGatesSim();
};

int lgs::LogicUnitBase::evaluate(void)
{
    switch (type) {
        case lgs::input: {
            return value;
        } break;

        case lgs::output: {
            return evaluate_input1();
        } break;

        case lgs::wire: {
            return evaluate_input1();
        } break;

        case lgs::notGate: {
            return !evaluate_input1();
        } break;

        case lgs::andGate: {
            return evaluate_input1() & evaluate_input2();
        } break;

        case lgs::orGate: {
            return evaluate_input1() | evaluate_input2();
        } break;

        case lgs::xorGate: {
            return evaluate_input1() ^ evaluate_input2();
        } break;
    }
    return 0;
}
