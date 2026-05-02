#pragma once

#include <chrono>

#include <alya/profiling/Print.hpp>

namespace alya::time {

template <typename unit>
const char* unitName();

template <>
const char* unitName<std::chrono::seconds>() { return "s"; }

template <>
const char* unitName<std::chrono::milliseconds>() { return "ms"; }

template <>
const char* unitName<std::chrono::microseconds>() { return "us"; }

template <>
const char* unitName<std::chrono::nanoseconds>() { return "ns"; }

using seconds = std::chrono::seconds;
using milliseconds = std::chrono::milliseconds;
using microseconds = std::chrono::microseconds;
using nanoseconds = std::chrono::nanoseconds;

using s = std::chrono::seconds;
using ms = std::chrono::milliseconds;
using us = std::chrono::microseconds;
using ns = std::chrono::nanoseconds;

}   //namespace alya::time

namespace alya {

/// @brief Timer with manuell timing or RAII
class Timer {
private:
    std::chrono::steady_clock::time_point startTimer;
    std::chrono::steady_clock::time_point endTimer;
    bool raiiVersion = true;

public:
    Timer() {
        startTimer = std::chrono::steady_clock::now();
    }

    ~Timer() {
        if(raiiVersion) {
            endTimer = std::chrono::steady_clock::now();
            auto clock = std::chrono::duration<double>(endTimer - startTimer).count();

            print("Time: ", clock, Endl{});
        }
    }

    void start() {
        startTimer = std::chrono::steady_clock::now();
    }

    void stop() {
        endTimer = std::chrono::steady_clock::now();
        raiiVersion = false;
    }

    double getTime() {
        auto end = (raiiVersion) ? std::chrono::steady_clock::now() : endTimer;

        return std::chrono::duration<double>(end - startTimer).count();
    }

    /// @brief Prints time
    /// @tparam unit time unit (e.g. alya::time::seconds, or standard std::chrono::seconds)
    /// @note if using namespace alya, short time acronym are available (e.g. alya::time::ms)
    template <typename unit = std::chrono::seconds>
    void printTime() const {
        auto end = (raiiVersion) ? std::chrono::steady_clock::now() : endTimer;

        using durationT = std::chrono::duration<double, typename unit::period>;     // duration<precision, ratio>
        auto clock = std::chrono::duration_cast<durationT>(end - startTimer).count();
        print("Time: ", clock, time::unitName<unit>(), Endl{});
    }

};

}   //namespace alya
