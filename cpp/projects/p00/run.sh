DIR="build"
if [ -d "$DIR" ]; then
  cd build && make && ./a.out
else
  mkdir build && cd build && cmake .. && make && ./a.out
fi
