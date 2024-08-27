#ifndef HUMAN_READABLE_HPP__
#define HUMAN_READABLE_HPP__

#include <cmath>
#include <cstdint>
#include <iostream>

namespace common {
/**
 * @brief Converts a size in bytes to a human-readable format.
 *
 * The `HumanReadable` struct provides a way to convert a size in bytes to a
 * more readable format, such as KB, MB, GB, etc., depending on the size.
 * It supports outputting the size in a concise format using the `<<` operator.
 *
 * @details This struct is useful when you want to display sizes in a format
 * that is easier to read and understand. For instance, instead of displaying
 * a size as "1048576 bytes," it can be displayed as "1MB (1048576 bytes)".
 */
struct HumanReadable {
  std::uintmax_t size{}; ///< The size in bytes.

  /**
   * @brief Constructs a `HumanReadable` object with a specified size.
   *
   * @param new_size The size in bytes to be converted to a human-readable format.
   *
   */
  HumanReadable(std::uintmax_t new_size) : size{new_size} {}

  /**
   * @brief Default destructor for `HumanReadable`.
   */
  ~HumanReadable() = default;

private:
  /**
   * @brief Outputs the size in a human-readable format.
   *
   * This friend function overloads the `<<` operator to output the size in a
   * human-readable format. The size is converted to the largest appropriate unit
   * (e.g., KB, MB, GB) and is displayed with one decimal place of precision.
   *
   * @param os The output stream.
   * @param hr The `HumanReadable` object to output.
   * @return A reference to the output stream.
   */
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

} // namespace common

#endif // HUMAN_READABLE_HPP__