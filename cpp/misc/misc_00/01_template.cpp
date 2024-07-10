
template <typename T>
struct AddOp {
  T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }

  static T run(T const &lhs, T const &rhs) {
    return (lhs + rhs + 20);
  }
};

template <class BFunc, int num = 3>
//template <typename BFunc> // This works too!!!
int mytemp(BFunc binop)
{
    return (binop(num, 10));
}

int main()
{
    return mytemp(AddOp<int>::run);
}