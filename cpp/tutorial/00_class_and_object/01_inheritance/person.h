#ifndef _PERSON_
#define PERSON

#include <string>
#include <iostream>

using namespace std;

class Person
{
    private:
        int  age;

    protected:
        string sex;

    public:
        Person();
        Person(string name, string sex, int age);

        string name;
 
        string get_sex();        
        int get_age();

        void set_sex(string);
        void set_age(int);
};




#endif