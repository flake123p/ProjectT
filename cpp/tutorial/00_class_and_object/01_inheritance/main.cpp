#include "student.h"
#include <iostream>

using namespace std;

int main()
{
    Person  person_0("Cindy", "Female", 20);
    Student student_0("Ben", "Male", 8, "0001", "1A");

    //person0
    cout << "Name : " << person_0.name << endl;
    cout << "Sex : "<< person_0.get_sex() << endl;
    cout << "Age : " << person_0.get_age() << endl;
    cout << endl;

    //student0
    cout << "Name : " << student_0.name << endl;
    cout << "Sex : "<< student_0.get_sex() << endl;
    cout << "Age : " << student_0.get_age() << endl;
    cout << "Student ID : " << student_0.get_student_id() << endl;
    cout << "Student Class : " << student_0.get_student_class() << endl;
    cout << endl;

    return 0;
}