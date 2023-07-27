#ifndef _STUDENT_
#define _STUDENT_

#include <iostream>
#include <string>

using namespace std;

class Student
{
    public:
        Student();        
        
        Student(string name, string sex, int age);
        
        string get_name();
        
        string get_sex();
        
        int get_age();

        void set_name(string name);

        void set_sex(string sex);

        void set_age(int age);

    private: 
        string name;        
        int age;        
        string sex;        
};

#endif