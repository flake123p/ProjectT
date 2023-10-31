#include <cstdlib>
#include <iostream>

using namespace std;

typedef struct student
{
    char name[20];
    int age;
    float score;
}Student;

bool compare_by_age(Student stu1, Student stu2)
{
    return stu1.age > stu2.age ? 1 : 0;
}

bool compare_by_score(Student stu1, Student stu2)
{
    return stu1.score > stu2.score ? 1 : 0;
}

void sort_students(Student *array, int n, bool(*p)(Student, Student))
{
    Student tmp;
    int flag = 0;
    for(int i = 0; (i < n-1) && (flag == 0); i++)
    {
        flag = 0;
        for(int j = 0; j < n-i-1; j++)
        {
            if(p(array[j], array[j+1]))
            {
                tmp = array[j];
                array[j] = array[j+1];
                array[j+1] = tmp;
                flag = 0;
            }
        }
    }
}

int main() 
{
    Student student_array[3] = {{"Tom", 19, 98},
                                {"Andy", 20, 60},
                                {"Mary", 21, 88}};
    
    //sort_students(student_array, 3, compare_by_age);
    sort_students(student_array, 3, compare_by_score);

    for(int i = 0; i < 3; i++)
        cout << student_array[i].name << " ";

    return 0;
}