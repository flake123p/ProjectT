#include <cstdlib>
#include <iostream>

using namespace std;

bool compare_greater(int number, int compare_number) {
    return number > compare_number;
}   

bool compare_less(int number, int compare_number) {
    return number < compare_number;
}

void compare_number_function(int *number_array, int count, int compare_number, bool (*p)(int, int)) {
    for (int i = 0; i < count; i++) {
        if (p(*(number_array + i), compare_number)) {
            cout << *(number_array + i) << " ";
        }
    }
    cout << endl;
}

int main() {

    int number_array[5] = {15, 34, 44, 56, 64};
    int compare_number = 50;

    compare_number_function(number_array, 5, compare_number, compare_greater);
    compare_number_function(number_array, 5, compare_number, compare_less);

    return 0;
}