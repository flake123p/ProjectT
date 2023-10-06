#pragma once

#include "LogicGatesSim.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <cassert>

#include <stdlib.h> /* rand */
#include <time.h>   /* time */

#define LGL_ASSERT assert

#define PRLOC printf("%s() %d\n", __func__, __LINE__);

/*
Encode every unit in 2D array for inter-connect index:

int col; // -1 for input
int row;
int col2; // -1 for input
int row2;

*/
namespace lgl {

struct LearnUnit {
    lgs::LogicUnitType type;
    bool is_this_unit_used;
    int cur_row;
    int cur_col;
    int in1_row;
    int in1_col; // -1 for input
    int in2_row;
    int in2_col; // -1 for input

    lgs::LogicUnitBase *lgsUnit;
};

class LearnGrid {
private:
    int _rows;
    int _cols;
public:
    using units_list_t = std::vector<lgs::LogicUnitBase *>;

    units_list_t input_list;

    int get_input_list_size()
    {
        return input_list.size();
    }
    void emplace_input_list(lgs::LogicUnitBase *u) {
        input_list.emplace_back(u);
    }
    
    struct LearnUnit *grid = nullptr;
    LearnGrid(int rows = 10, int cols = 10) : _rows{rows}, _cols{cols} {
        grid = (struct LearnUnit *)malloc(sizeof(struct LearnUnit) * _rows * _cols);
        struct LearnUnit * curr = grid;
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                curr->cur_col = c;
                curr->cur_row = r;
                curr->lgsUnit = new lgs::LogicUnitBase;
                curr->is_this_unit_used = false;
                curr++;
            }
        }
    }
    ~LearnGrid() {
        if (grid == nullptr) {
            return;
        }
        struct LearnUnit * curr = grid;
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                delete(curr->lgsUnit);
                curr++;
            }
        }
        free(grid);
        grid = nullptr;
    }
    struct LearnUnit *get_unit(int rows, int cols) {
        assert(cols < _cols && cols >= 0);
        assert(rows < _rows && rows >= 0);
        int index = rows * _cols + cols;
        return grid + index;
    }
    lgs::LogicUnitBase *get_lgs_unit(int rows, int cols) {
        if (cols == -1) {
            assert(rows < get_input_list_size() && rows >= 0);
            return input_list[rows];
        } else {
            assert(cols < _cols && cols >= 0);
            assert(rows < _rows && rows >= 0);
            return get_unit(rows, cols)->lgsUnit;
        }
    }
    void calculate_cols_rows_by_index(int index, int &out_rows, int &out_cols) {
        out_rows = index / _rows;
        out_cols = index % _rows;
    }
    void calculate_cols_rows_by_index_column_first(int index, int &out_rows, int &out_cols) {
        out_rows = index % _rows;
        out_cols = index / _rows;
    }
    void randomly_initialize_types() {
        int x;
        // not, and, or, invalid
        struct LearnUnit * curr = grid;
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                x = rand() % 3;
                switch (x) {
                    case 0: curr->type = lgs::notGate; break;
                    case 1: curr->type = lgs::andGate; break;
                    case 2: curr->type = lgs::orGate; break;
                    //case 3: curr->type = lgs::invalid_type; break;
                    default: assert(0); break;
                }
                curr++;
            }
        }
    };
    void randomly_initialize_connection(struct LearnUnit * curr) {
        assert(get_input_list_size() != 0);

        const int MAX_LAYERS_AHEAD = 3;

        bool choose_input = false;
        int layers_ahead = curr->cur_col;
        int min, max;
        int max_real; // include inputs

        if (layers_ahead < MAX_LAYERS_AHEAD) {
            choose_input = true;
            min = 0;
            max = _rows * layers_ahead;
            max_real = max + get_input_list_size();
        } else {
            min = _rows * (layers_ahead - MAX_LAYERS_AHEAD);
            max = _rows * (layers_ahead);
            max_real = max;
        }
        int rand_interval = max_real - min;
        assert(rand_interval != 0);
        int in1_index = 0, in2_index = 0;
        {
            int c,r;
            struct LearnUnit *choosed;
            //
            // do input 1 first
            //
            do {
                in1_index = rand() % rand_interval + min;
                if (in1_index >= max) { // input choosed
                    curr->in1_col = -1;
                    curr->in1_row = in1_index - max;
                } else {
                    calculate_cols_rows_by_index_column_first(in1_index, r, c);
                    choosed = get_unit(r, c);
                    if (choosed->type == lgs::invalid_type) {
                        continue;
                    } else {
                        curr->in1_row = r;
                        curr->in1_col = c;
                        if (curr->in1_col >= curr->cur_col) {
                            printf("in:%d, curr:%d\n", curr->in1_col, curr->cur_col);
                            assert(0);
                        }
                    }
                }
            } while (false);
            //
            // do input 2
            //
            auto rand2_lambda = [&]() -> int { 
                int random_result;
                while (true) {
                    random_result = rand() % rand_interval + min;
                    if (random_result != in1_index)
                        break;
                }
                return random_result; 
            };
            if (curr->type & lgs::binary_type) {
                do {
                    in2_index = rand2_lambda();
                    
                    if (in2_index >= max) { // input choosed
                        curr->in2_col = -1;
                        curr->in2_row = in2_index - max;
                    } else {
                        calculate_cols_rows_by_index_column_first(in2_index, r, c);
                        choosed = get_unit(r, c);
                        if (choosed->type == lgs::invalid_type) {
                            continue;
                        } else {
                            curr->in2_row = r;
                            curr->in2_col = c;
                            if (curr->in2_col >= curr->cur_col) {
                                printf("in:%d, curr:%d\n", curr->in2_col, curr->cur_col);
                                assert(0);
                            }
                        }
                    }
                } while (false);
            }
        }
    }
    void randomly_initialize_connections() {
        //
        // choose any one in 3 layers ahead
        //
        assert(grid != nullptr);
        struct LearnUnit * curr = grid;
        for (int i = 0; i < _rows * _cols; i++) {
            randomly_initialize_connection(curr);
            curr++;
        }
    };
    void initialize_lgs_grid()
    {
        struct LearnUnit *learn_unit;
        lgs::LogicUnitBase *lgs_unit, *in1, *in2;
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                learn_unit = get_unit(r, c);
                lgs_unit = get_lgs_unit(r, c);

                lgs_unit->type = learn_unit->type;
                in1 = get_lgs_unit(learn_unit->in1_row, learn_unit->in1_col);

                if (lgs_unit->type & lgs::unary_type) {
                    lgs_unit->import(*in1);
                } else {
                    in2 = get_lgs_unit(learn_unit->in2_row, learn_unit->in2_col);
                    lgs_unit->import(*in1, *in2);
                }
            }
        }
    }
    static void random_init() {
        srand( time(NULL) );
    }
    void dump() {
        lgs::LogicUnitBase dummy;
        struct LearnUnit * curr = grid;
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                printf("[%2d/%2d/ %7s_%d]", curr->cur_row, curr->cur_col, dummy.type2str(curr->type).c_str(), curr->is_this_unit_used);
                printf("<%2d,%2d>", curr->in1_row, curr->in1_col);
                if (curr->type & lgs::unary_type) {
                    printf("<--, --> ");
                } else {
                    printf("<%2d, %2d> ", curr->in2_row, curr->in2_col);
                }
                curr++;
            }
            printf("\n");
        }
        printf("\n");
    }
    void traverse_used_unit(struct LearnUnit * curr) {
        curr->is_this_unit_used = true;
        if (curr->in1_col != -1) {
            traverse_used_unit(get_unit(curr->in1_row, curr->in1_col));
        }
        if (curr->type & lgs::binary_type) {
            if (curr->in2_col != -1) {
                traverse_used_unit(get_unit(curr->in2_row, curr->in2_col));
            }
        }
    }
    void unset_all_used_unit() {
        struct LearnUnit * curr = grid;
        for (int i = 0; i < _rows * _cols; i++) {
            curr->is_this_unit_used = false;
            curr++;
        }
    }
};

void LogicGatesLearn_Demo();
void LogicGatesLearn_DemoBruteForce();

} //namespace lgl
