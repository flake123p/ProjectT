
#include "all.hpp"

/*
    Ref: https://en.cppreference.com/w/cpp/language/reinterpret_cast

        reinterpret_cast : Converts between types by reinterpreting the underlying bit pattern.

    Ref: https://learn.microsoft.com/zh-tw/cpp/cpp/reinterpret-cast-operator?view=msvc-170

        Usage:
            pointer to pointer
            pointer to integer ................................................................. KEYPOINT
            integer to pointer ................................................................. KEYPOINT
*/

int main() 
{
    {
        int i = 7;
    
        // pointer to integer and back
        std::uintptr_t v1 = reinterpret_cast<std::uintptr_t>(&i); // static_cast is an error
        std::cout << "The value of &i is " << std::showbase << std::hex << v1 << '\n';
        int* p1 = reinterpret_cast<int*>(v1);
        assert(p1 == &i);
    }
    {
        uint32_t buf32[10] = {0};
        uint8_t *buf8;

        uint64_t address = reinterpret_cast <uint64_t>( buf32 );

        buf8 = reinterpret_cast <uint8_t *>( buf32 );
        buf8[0] = 0x11;
        buf8[1] = 0x22;
        printf("buf32[0] = 0x%08x\n", buf32[0]);
        
        std::cout << std::hex << "address = 0x" << address << "\n";
    }

    return 0;
}
