#ifndef HUMAN_READABLE_HPP__
#define HUMAN_READABLE_HPP__

#include <cmath>
#include <cstdint>
#include <iostream>

namespace common {
struct HumanReadable {
  std::uintmax_t size{};
  HumanReadable(std::uintmax_t new_size) : size{new_size} {}
  ~HumanReadable() = default;

 private:
  friend std::ostream &operator<<(std::ostream &os, HumanReadable hr) {
    int i{};
    double mantissa = hr.size;
    for (; mantissa >= 1000.; mantissa /= 1000., ++i) {
    }
    mantissa = std::ceil(mantissa * 10.) / 10.;
    os << mantissa << "BKMGTPE"[i];
    return i == 0 ? os : os << "B (" << hr.size << ")";
  }
};
}  // namespace common

#endif  // HUMAN_READABLE_HPP__