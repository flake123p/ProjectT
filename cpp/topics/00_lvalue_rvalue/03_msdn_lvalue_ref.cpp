
#include "all.hpp"

//
// https://learn.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2010/w7049scy(v=vs.100)
//

/*
    type-id & cast-expression

    A reference must be initialized and cannot be changed.
*/

using namespace std;
int main_ISO_Cpp();

struct Person
{
    char* Name;
    short Age;
};

int main()
{
   // Declare a Person object.
   Person myFriend;

   // Declare a reference to the Person object.
   Person& rFriend = myFriend;

   // Set the fields of the Person object.
   // Updating either variable changes the same object.
   myFriend.Name = "Bill";                    //ISO C++ forbids converting a string constant to ‘char*’
   rFriend.Age = 40;

   // Print the fields of the Person object to the console.
   cout << rFriend.Name << " is " << myFriend.Age << endl;

   main_ISO_Cpp();
   return 0;
}

int main_ISO_Cpp()
{
   // Declare a Person object.
   Person myFriend;

   // Declare a reference to the Person object.
   Person& rFriend = myFriend;

   // Set the fields of the Person object.
   // Updating either variable changes the same object.
   char _[] = "Bill";
   myFriend.Name = _;
   rFriend.Age = 40;

   // Print the fields of the Person object to the console.
   cout << rFriend.Name << " is " << myFriend.Age << endl;

   return 0;
}