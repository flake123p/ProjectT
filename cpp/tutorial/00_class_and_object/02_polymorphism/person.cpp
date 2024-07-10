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

string Person::get_name()
{
    return name;
}

string Person::get_sex()
{
    return sex;
}

int Person::get_age()
{
    return age;
}

void Person::set_name(string name)
{
    this->name = name;
}

void Person::set_sex(string sex)
{
    this->sex = sex;
}

void Person::set_age(int age)
{
    this->age = age;
}

void Person::print_information()
{
    cout << "Name : " << this->name << endl;
    cout << "Sex : "<< this->sex << endl;
    cout << "Age : " << this->age << endl;
}

