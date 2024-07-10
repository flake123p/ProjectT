#pragma once

#include <string>
#include <unordered_map>
#include <cassert>

#define LGS_ASSERT assert

namespace lgs {

enum LogicUnitType {
    invalid_type,

    unary_type = 0x100,
    input,          // unary
    output,         // unary
    wire,           // unary
    notGate,        // unary

    binary_type = 0x200,
    andGate,        // binary
    orGate,         // binary
    xorGate,        // binary
   
};
#define _TYPE2STR(a) case a: return std::string(#a);
#define TYPE2STR \
    _TYPE2STR(input) \
    _TYPE2STR(output) \
    _TYPE2STR(wire) \
    _TYPE2STR(notGate) \
    _TYPE2STR(andGate) \
    _TYPE2STR(orGate) \
    _TYPE2STR(xorGate)

class LogicUnitBase {
public:
    LogicUnitBase(LogicUnitType initType = input, int initValue = 0) : type{initType}, value{initValue} {};
    LogicUnitType type;
    int value = 0;
    LogicUnitBase *in = nullptr;
    LogicUnitBase *in2 = nullptr;
    std::string name = "";
    int evaluate(void);
    int evaluate_input1(void) {
        if (in == nullptr) {
            LGS_ASSERT(0);
            return -1;
        } else {
            return in->evaluate();
        }
    };
    int evaluate_input2(void) {
        if (in2 == nullptr) {
            LGS_ASSERT(0);
            return -1;
        } else {
            return in2->evaluate();
        }
    };
    void import(LogicUnitBase &_in) {in = &_in;};
    void import(LogicUnitBase &_in, LogicUnitBase &_in2) {in = &_in;in2 = &_in2;};
    std::string type2str(LogicUnitType _type = invalid_type) {
        if (_type == invalid_type)
            _type = this->type;

        switch (_type) {
            TYPE2STR
            default:
                return "UNKNOWN_TYPE";
        }
    }
    void type_dump(int level = 0) {
        for (int i = 0; i < level; i ++) {
            printf("    ");
        }
        auto full_name = type2str() + name;
        printf("%s\n", full_name.c_str());
        if (in != nullptr) {
            in->type_dump(level+1);
        }
        if (in2 != nullptr && (type & binary_type)) {
            in2->type_dump(level+1);
        }
    }
};

}
