#ifndef _STUDENT_
#define _STUDENT_

#include <iostream>
#include <string>

using namespace std;

class Student
{
    public:
        Student();        
        
        //Student(string name, string sex, int age):name(name), sex(sex), age(age) {};  //等於下一行和他的函式加起來的功能
        Student(string name, string sex, int age, string student_class);
                
        string get_sex();
        
        int get_age();

        string get_student_class();

        void set_sex(string sex);

        void set_age(int age);

        void set_student_class(string student_class);

        string name;  

    protected:
        string student_class;

    private: 
        int age;        
        string sex;        
};

#endif