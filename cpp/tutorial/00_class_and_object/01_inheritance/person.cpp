#include "person.h"

Person::Person()
{
    ;
}

Person::Person(string name, string sex, int age)
{
    this->name = name;
    this->sex = sex;
    this->age = age;
}

string Person::get_sex()
{
    return sex;
}

int Person::get_age()
{
    return age;
}

void Person::set_sex(string sex)
{
    this->sex = sex;
}

void Person::set_age(int age)
{
    this->age = age;
}

