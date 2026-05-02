#pragma once

#include <iomanip>
#include <iostream>

namespace alya {

struct Endl {};
struct Fixed { int precision; };
struct Sci { int precision; };

}   //namespace alya

namespace alya::internal {
    
template <typename... T>
void printT(std::ostream& stream, T&&... args) {
    auto flags = stream.flags();
    auto prec = stream.precision();

    auto out = [&stream](auto& x) {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Endl>) {
            stream << std::endl;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Fixed>) {
            stream << std::fixed << std::setprecision(x.precision);
        } else if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Sci>) {
            stream << std::scientific << std::setprecision(x.precision);
        } else {
            stream << x;
        }
    };

    (out(args), ...);

    stream.flags(flags);
    stream.precision(prec);
}

}   //namespace alya::internal

namespace alya {

template <typename... T>
void print(T&&... args) { internal::printT(std::cout, args...); }

template <typename... T>
void eprint(T&&... args) { internal::printT(std::cerr, args...); }

}   //namespace alya
