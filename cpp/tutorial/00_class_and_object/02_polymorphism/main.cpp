#include "student.h"
#include <iostream>

using namespace std;

int main()
{
    Person  person_0("Cindy", "Female", 20);
    Student student_0("Ben", "Male", 8, "0001", "1A");

    //person0
    person_0.print_information();
    cout << endl;

    //student0
    student_0.print_information();
    cout << endl;

    return 0;
}