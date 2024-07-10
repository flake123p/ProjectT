#ifndef _STUDENT_
#define _STUDENT_

#include "person.h"
#include <iostream>
#include <string>

using namespace std;

class Student:public Person
{
    public:
        //overload
        Student();               
        Student(string name, string sex, int age, string student_id, string student_class);

        string get_student_id();
        string get_student_class();
        void set_student_id(string student_id);
        void set_student_class(string student_class);

        //override
        void print_information();

    private: 
        string student_id;
        string student_class;
};

#endif