
#pragma once

//
//  Obtain the number of elements in the given C array
//
// #define GET_ARRAY_LEN(arrayName)  (sizeof( arrayName ) / sizeof(( arrayName)[ 0 ] ))
// #define ARRAY_AND_SIZE(arrayName) (arrayName),GET_ARRAY_LEN(arrayName)
// #define ARRAY_SIZE              GET_ARRAY_LEN
// #define ARRAY_LENGTH            GET_ARRAY_LEN
// #define ARRAY_LEN               GET_ARRAY_LEN
// #define LENGTH_OF_ARRAY         GET_ARRAY_LEN
// #define LEN_OF_ARRAY            GET_ARRAY_LEN


/*
        Array Size C++ Version

        https://stackoverflow.com/questions/65475688/understanding-expression-does-not-compute-the-number-of-elements-in-this-array
*/
template<typename T, size_t N>
constexpr size_t size_of_array( T (&_arr)[N]) {
    (void)_arr;
    return N;
}
