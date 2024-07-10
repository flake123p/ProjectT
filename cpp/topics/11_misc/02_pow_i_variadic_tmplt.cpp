#include "_float.h"
#include "limits.h"
#include <iostream>       // std::cout
#include <thread>         // std::thread, std::chrono
#include <vector>
#include "FunctionTraits.h"

double my_pow_uint(double x, unsigned int n)
{
  double value = 1.0;

  /* repeated squaring method 
   * returns 0.0^0 = 1.0, so continuous in x
   */
  while (n) {
    value *= x;
    n--;
  }

  return value;
}

double gsl_pow_uint(double x, unsigned int n)
{
  double value = 1.0;

  /* repeated squaring method 
   * returns 0.0^0 = 1.0, so continuous in x
   */
  do {
    if(n & 1) value *= x;  /* for n odd */
    n >>= 1;
    x *= x;
  } while (n);

  return value;
}

double gsl_pow_int(double x, int n)
{
  unsigned int un;

  if(n < 0) {
    x = 1.0/x;
    un = -n;
  } else {
    un = n;
  }

  return gsl_pow_uint(x, un);
}

// powi_impl()
double torch_powi(double a, int b) // Flake: only for positive exponent!!
{
  double result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}
/*
Flake: Return weired when exponent is negative !!!

static inline HOST_DEVICE T powi(T a, T b) {
  if ( b < 0 ) {
      if ( a == 1 ) {
          return 1;
      } else if ( a == -1 ) {
          auto negative = (-b) % static_cast<T>(2);
          return negative ? -1 : 1;
      } else {
          return 0;
      }
  }
  return powi_impl(a, b);
}
*/

#define INLINE_FUN
INLINE_FUN double gsl_pow_2(const double x) { return x*x;   }
INLINE_FUN double gsl_pow_3(const double x) { return x*x*x; }
INLINE_FUN double gsl_pow_4(const double x) { double x2 = x*x;   return x2*x2;    }
INLINE_FUN double gsl_pow_5(const double x) { double x2 = x*x;   return x2*x2*x;  }
INLINE_FUN double gsl_pow_6(const double x) { double x2 = x*x;   return x2*x2*x2; }
INLINE_FUN double gsl_pow_7(const double x) { double x3 = x*x*x; return x3*x3*x;  }
INLINE_FUN double gsl_pow_8(const double x) { double x2 = x*x;   double x4 = x2*x2; return x4*x4; }
INLINE_FUN double gsl_pow_9(const double x) { double x3 = x*x*x; return x3*x3*x3; }

#define FUNC_AND_NAME(f) f,#f

template<typename Func_T>
void time_prof(Func_T func, const char *id)
{
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    start = std::chrono::steady_clock::now();
    {
        func();
    }
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    double t = elapsed_seconds.count() * 1000; // t number of seconds, represented as a `double`
    printf("t = %13.6f ms [%s]\n", t, id);
}

template<typename Func_T, typename... Args>
void loop_tester(Func_T func, volatile int &loops, volatile const Args... args) {
    double result;
    for (int i = 0; i < loops; i++) {
        result = func(args...);
    }
    (void)result;
} 

int main() 
{
    volatile int loops = 100000000;
    volatile double input = 2.1;
    volatile int expo = 7;

    time_prof (
        [&]() -> void {
            loop_tester(gsl_pow_7, loops, input);
        }, 
        "gsl_pow_7"
    );
    time_prof (
        [&]() -> void {
            loop_tester(gsl_pow_int, loops, input, expo);
        }, 
        "gsl_pow_int"
    );
    time_prof (
        [&]() -> void {
            loop_tester(gsl_pow_uint, loops, input, expo);
        }, 
        "gsl_pow_uint"
    );
    time_prof (
        [&]() -> void {
            loop_tester(my_pow_uint, loops, input, expo);
        }, 
        "my_pow_uint"
    );

    // printf("%f\n", my_pow_uint(2.1, 7));
    // printf("%f\n", gsl_pow_uint(2.1, 7));

    BASIC_ASSERT(my_pow_uint(2, 7) == gsl_pow_uint(2, 7));

    time_prof (
        [&]() -> void {
            loop_tester(torch_powi, loops, input, expo);
        }, 
        "torch_powi"
    );

    // printf("%f\n", gsl_pow_int(2.1, 7));
    // printf("%f\n", torch_powi(2.1, 7));
    
    return 0;
}