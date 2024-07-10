//
// How to interpret complex C/C++ declarations
//      https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
//

#include "all.hpp"

int main() 
{
    {
        int (*p)(char); // This declares p as a pointer to a function that takes a char argument and returns an int.
    }
    {
        char ** (*p)(float, float); // takes two floats and returns a pointer to a pointer to a char
    }
    {
        void * (*a[5])(char * const, char * const); // an array of 5 pointers to functions
    }
    /*
        The right-left rule [Important]

        1. Start reading the declaration from the innermost parentheses.

        2. go right

        3. go left

        4. When you encounter parentheses, the direction should be reversed.

        5. Once everything in the parentheses has been parsed, jump out of it. Continue till the whole declaration has been parsed.

        One small change to the right-left rule: 
        When you start reading the declaration for the first time, 
        you have to start from the identifier, and not the innermost parentheses.
    */
    {
        int * (* (*fp1) (int) ) [10];
        /*
            1. Start from the variable name -------------------------- fp1
            2. Nothing to right but ) so go left to find * ----------- is a pointer
            3. Jump out of parentheses and encounter (int) ----------- to a function that takes an int as argument
            4. Go left, find * --------------------------------------- and returns a pointer
            5. Jump put of parentheses, go right and hit [10] -------- to an array of 10
            6. Go left find * ---------------------------------------- pointers to
            7. Go left again, find int ------------------------------- ints.
        */
    }
    {
        int *( *( *arr[5])())();
        /*
            1. Start from the variable name --------------------- arr
            2. Go right, find array subscript ------------------- is an array of 5
            3. Go left, find * ---------------------------------- pointers
            4. Jump out of parentheses, go right to find () ----- to functions
            5. Go left, encounter * ----------------------------- that return pointers
            6. Jump out, go right, find () ---------------------- to functions
            7. Go left, find * ---------------------------------- that return pointers
            8. Continue left, find int -------------------------- to ints.
        */
    }

    {
        float ( * ( *b()) [] )();           // b is a function that returns a 
                                            // pointer to an array of pointers
                                            // to functions returning floats.

        void * ( *c) ( char, int (*)());    // c is a pointer to a function that takes
                                            // two parameters:
                                            //     a char and a pointer to a
                                            //     function that takes no
                                            //     parameters and returns
                                            //     an int
                                            // and returns a pointer to void.

        void ** (*d) (int &, char **(*)(char *, char **));        
                                            // d is a pointer to a function that takes
                                            // two parameters:
                                            //     a reference to an int and a pointer
                                            //     to a function that takes two parameters:
                                            //        a pointer to a char and a pointer
                                            //        to a pointer to a char
                                            //     and returns a pointer to a pointer 
                                            //     to a char
                                            // and returns a pointer to a pointer to void

        float ( * ( * e[10]) (int &) ) [5];
                                            // e is an array of 10 pointers to 
                                            // functions that take a single
                                            // reference to an int as an argument 
                                            // and return pointers to
                                            // an array of 5 floats.
    }

    return 0;
}
