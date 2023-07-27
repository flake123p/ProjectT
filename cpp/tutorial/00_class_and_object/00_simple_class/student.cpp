#include "student.h"

Student::Student()
{
    ;
}

Student::Student(string name, string sex, int age)
{
    this->name = name;
    this->sex = sex;
    this->age = age;
}

string Student::get_name()
{
    return name;
}

string Student::get_sex()
{
    return sex;
}

int Student::get_age()
{
    return age;
}

void Student::set_name(string name)
{
    this->name = name;
}

void Student::set_sex(string sex)
{
    this->sex = sex;
}

void Student::set_age(int age)
{
    this->age = age;
}

