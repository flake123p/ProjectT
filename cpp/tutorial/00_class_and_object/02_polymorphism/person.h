#ifndef _PERSON_
#define PERSON

#include <string>
#include <iostream>

using namespace std;

class Person
{
    private:
        string name;
        string sex;
        int  age;

    public:
        Person();
        Person(string name, string sex, int age);

        string get_name();        
        string get_sex();        
        int get_age();

        void set_name(string);
        void set_sex(string);
        void set_age(int);

        void print_information();    
};




#endif