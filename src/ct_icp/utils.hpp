#ifndef CT_ICP_UTILS_HPP
#define CT_ICP_UTILS_HPP

#if CT_ICP_CPP_STANDARD == 17
#include <filesystem>
// #include <experimental/filesystem>
namespace fs = std::filesystem;
//namespace fs = std::experimental::filesystem;
#define WITH_STD_FILESYSTEM 1
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define CT_ICP_IS_WINDOWS
#endif



#endif //CT_ICP_UTILS_HPP
