
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <climits>
#include <cfloat>
#include <variant>
#include <optional>
#include <any>
#include <filesystem>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define TSIZE(a) std::cout << "sizeof(" #a ") = " << sizeof(a) << std::endl
#define TNAME(a) std::cout << "typeid(" #a ").name() = " << typeid(a).name() << std::endl
#define CDUMP(a) COUT(a);TSIZE(a);TNAME(a)
#define PRINT_FUNC printf("%s()\n", __func__);

/*
  https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md

  std::filesystem

  The new std::filesystem library provides a standard way to manipulate files, directories, and paths in a filesystem.

  Here, a big file is copied to a temporary path if there is available space:
*/

int main() 
{
  const auto bigFilePath {"bigFileToCopy"};

  if (std::filesystem::exists(bigFilePath)) {
    const auto bigFileSize {std::filesystem::file_size(bigFilePath)};
    std::filesystem::path tmpPath {"/tmp"};
    if (std::filesystem::space(tmpPath).available > bigFileSize) {
      std::filesystem::create_directory(tmpPath.append("example"));
      std::filesystem::copy_file(bigFilePath, tmpPath.append("newFile"));
    }
  } else {
    printf("file not exist...\n");
  }

  return 0;
}
