
#include "all.hpp"

//
// https://learn.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2010/dd293665(v=vs.100)
//

// MemoryBlock.h
//#pragma once
#include <iostream>
#include <algorithm>

class MemoryBlock
{
public:

   // Simple constructor that initializes the resource.
   explicit MemoryBlock(size_t length)
      : _length(length)
      , _data(new int[length])
   {
      std::cout << "In MemoryBlock(size_t). length = "
                << _length << "." << std::endl;
   }

   // Destructor.
   ~MemoryBlock()
   {
      std::cout << "In ~MemoryBlock(). length = "
                << _length << ".";
      
      if (_data != NULL)
      {
         std::cout << " Deleting resource.";
         // Delete the resource.
         delete[] _data;
      }

      std::cout << std::endl;
   }

   // Copy constructor.
   MemoryBlock(const MemoryBlock& other)
      : _length(other._length)
      , _data(new int[other._length])
   {
      std::cout << "In MemoryBlock(const MemoryBlock&). length = " 
                << other._length << ". Copying resource." << std::endl;

      std::copy(other._data, other._data + _length, _data);
   }

   // Copy assignment operator.
   MemoryBlock& operator=(const MemoryBlock& other)
   {
      std::cout << "In operator=(const MemoryBlock&). new length = " 
                << other._length << ". Copying resource." 
                << " old length = " << _length
                << std::endl;

      if (this != &other)
      {
         // Free the existing resource.
         delete[] _data;

         _length = other._length;
         _data = new int[_length];
         std::copy(other._data, other._data + _length, _data);
      }
      return *this;
   }

   // Move constructor.
   MemoryBlock(MemoryBlock&& other)
      : _data(NULL)
      , _length(0)
   {
      std::cout << "In MemoryBlock(MemoryBlock&&). length = " 
               << other._length << ". Moving resource." << std::endl;

      // Copy the data pointer and its length from the 
      // source object.
      _data = other._data;
      _length = other._length;

      // Release the data pointer from the source object so that
      // the destructor does not free the memory multiple times.
      other._data = NULL;
      other._length = 0;
   }

   // Move assignment operator.
   MemoryBlock& operator=(MemoryBlock&& other)
   {
      std::cout << "In operator=(MemoryBlock&&). length = " 
               << other._length << "." << std::endl;

      if (this != &other)
      {
         // Free the existing resource.
         delete[] _data;

         // Copy the data pointer and its length from the 
         // source object.
         _data = other._data;
         _length = other._length;

         // Release the data pointer from the source object so that
         // the destructor does not free the memory multiple times.
         other._data = NULL;
         other._length = 0;
      }
      return *this;
   }

   // Retrieves the length of the data resource.
   size_t Length() const
   {
      return _length;
   }

private:
   size_t _length; // The length of the resource.
   int* _data; // The resource.
};

using namespace std;

int main()
{
    {
        printf("\n Basic Exmaple (Start):\n");

        MemoryBlock mb1 = MemoryBlock(100);
        MemoryBlock mb2 = MemoryBlock(20);
        mb1 = mb2;

        printf("\n Basic Exmaple (End):\n");
    }
    {
        printf("\n Move Exmaple (Start):\n");

        MemoryBlock mb1 = MemoryBlock(100);
        MemoryBlock mb2 = MemoryBlock(20);
        mb1 = std::move(mb2);

        printf("\n Move Exmaple (End):\n");
    }
    {
        printf("\n Vector Exmaple (Start):\n");

        // Create a vector object and add a few elements to it.
        vector<MemoryBlock> v;

        printf(">>> push 1st start ...\n");
        v.push_back(MemoryBlock(25));
        printf(">>> push 1st end   ...\n");

        printf(">>> push 2nd start ...\n");
        v.push_back(MemoryBlock(75));
        printf(">>> push 2nd end   ...\n");

        printf(">>> push 3rd start ...\n");
        v.push_back(MemoryBlock(88));
        printf(">>> push 3rd end   ...\n");

        printf("\n Vector Exmaple (End):\n");
    }
    {
        printf("\n Vector Exmaple MOVE (Start):\n");

        // Create a vector object and add a few elements to it.
        vector<MemoryBlock> v;

        printf(">>> push 1st start ...\n");
        v.push_back(std::move(MemoryBlock(25)));
        printf(">>> push 1st end   ...\n");

        printf(">>> push 2nd start ...\n");
        v.push_back(std::move(MemoryBlock(75)));
        printf(">>> push 2nd end   ...\n");

        printf(">>> push 3rd start ...\n");
        v.push_back(std::move(MemoryBlock(88)));
        printf(">>> push 3rd end   ...\n");

        printf("\n Vector Exmaple MOVE (End):\n");
    }
    return 0;
}
