#include "student.h"

Student::Student()
{
    ;
}

Student::Student(string name, string sex, int age, string student_id, string student_class)
{
    this->name = name;
    this->set_sex(sex);
    this->set_age(age);
    this->student_id = student_id;
    this->student_class = student_class;
}
string Student::get_student_id()
{
    return student_id;
}

string Student::get_student_class()
{
    return student_class;
}


