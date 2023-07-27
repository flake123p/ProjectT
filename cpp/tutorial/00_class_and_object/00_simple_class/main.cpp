#include "student.h"
#include <iostream>

using namespace std;

int main()
{
    Student student_0("Ben", "Male", 8);
    Student student_1;

    //student0
    cout << "Name : " << student_0.get_name() << endl;
    cout << "Sex : "<< student_0.get_sex() << endl;
    cout << "Age : " << student_0.get_age() << endl;
    cout << endl;

    //student1
    student_1.set_name("Amy");
    student_1.set_sex("Female");
    student_1.set_age(12);
    cout << "Name : " << student_1.get_name() << endl;
    cout << "Sex : "<< student_1.get_sex() << endl;
    cout << "Age : " << student_1.get_age() << endl;
    cout << endl;

    return 0;
}