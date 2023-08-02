#include "student.h"
#include <iostream>

using namespace std;

int main()
{
    Student student_0("Ben", "Male", 8);                //隱性
    //Student student_0 = Student("Ben", "Male", 8);    //顯性
    Student student_1;

    Student *student_2 = new Student("Lose", "Female", 18);//物件指標
    //Student *student_2("Tom", "Male", 13);            //用隱性的寫法去建一個物件的指標會失敗

    Student *student_3;

    //student0
    cout << "Name : " << student_0.get_name() << endl;
    cout << "Sex : "<< student_0.get_sex() << endl;
    cout << "Age : " << student_0.get_age() << endl;
    cout << endl;

    //student1
    student_1.set_name("Amy");
    student_1.set_sex("Female");
    student_1.set_age(12);
    cout << "Name : " << student_1.get_name() << endl;
    cout << "Sex : "<< student_1.get_sex() << endl;
    cout << "Age : " << student_1.get_age() << endl;
    cout << endl;

    //student2
    cout << "Name : " << student_2->get_name() << endl;
    cout << "Sex : "<< student_2->get_sex() << endl;
    cout << "Age : " << student_2->get_age() << endl;
    cout << endl;

    //student3  無法使用，必須先將指標指向某個student物件
    /*student_3->set_name("Jack");
    student_3->set_sex("Male");
    student_3->set_age(17);
    cout << "Name : " << student_3->get_name() << endl;
    cout << "Sex : "<< student_3->get_sex() << endl;
    cout << "Age : " << student_3->get_age() << endl;
    cout << endl;*/

    return 0;
}