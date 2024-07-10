#include "student.h"

Student::Student()
{
    ;
}

Student::Student(string name, string sex, int age, string student_class)
{
    this->name = name;
    this->sex = sex;
    this->age = age;
    this->student_class = student_class;
}

string Student::get_sex()
{
    return sex;
}

int Student::get_age()
{
    return age;
}

string Student::get_student_class()
{
    return student_class;
}

void Student::set_sex(string sex)
{
    this->sex = sex;
}

void Student::set_age(int age)
{
    this->age = age;
}

void Student::set_student_class(string student_class)
{
    this->student_class = student_class;
}

