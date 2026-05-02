#pragma once

#include <cstdlib>
#include <string>

#include <alya/profiling/Print.hpp>

namespace alya {

inline void errorMessage(const char* func, const char* file, int line, const std::string& msg) {
    eprint("Error in ", func, " (", file, ":", line, "): ", msg, "\n");
    std::exit(1);
}

}   //namespace alya

#define ERRORMESSAGE(msg) errorMessage(__func__, __FILE__, __LINE__, msg);
