#ifndef CMDLINE_OPTS_HPP_
#define CMDLINE_OPTS_HPP_

#include <boost/program_options.hpp>
#include <cstdint>
#include <string>

namespace po = boost::program_options;

namespace supertensor {
namespace gputucker {

/**
 * @brief Command line options for the Tucker decomposition program.
 * 
 * This class defines and handles the command line options used by the Tucker decomposition
 * program. It parses command line arguments and stores essential parameters such as the
 * input path, tensor order, rank, and the number of GPUs to be used.
 * 
 * @author Jihye Lee
 * @date 2023-08-10
 * @version 1.0.0
 */
class CommandLineOptions {
 public:
  /**
   * @brief Enum to represent the status of command line option parsing.
   */
  enum ReturnStatus { 
    OPTS_SUCCESS, ///< Parsing was successful.
    OPTS_HELP,    ///< Help was requested.
    OPTS_FAILURE  ///< Parsing failed.
  };

  /**
   * @brief Constructor for CommandLineOptions.
   * 
   * Initializes the command line options with default values.
   */
  CommandLineOptions();

  /**
   * @brief Destructor for CommandLineOptions.
   */
  ~CommandLineOptions();

  /**
   * @brief Parses the command line arguments.
   * 
   * This function parses the command line arguments and stores the values for input path,
   * order, rank, and GPU count based on the provided arguments.
   * 
   * @param argc The argument count.
   * @param argv The argument vector.
   * @return A status code indicating the result of parsing.
   */
  ReturnStatus Parse(int argc, char *argv[]);

  /**
   * @brief Retrieves the input path specified in the command line options.
   * 
   * @return A constant reference to the input path string. Returns an empty string if no
   *         input path is specified.
   */
  const std::string& get_input_path() const;

  /**
   * @brief Retrieves the order of the tensor.
   * 
   * @return The order of the tensor.
   */
  inline int get_order() { return this->_order; }

  /**
   * @brief Retrieves the rank for Tucker decomposition.
   * 
   * @return The rank for the decomposition.
   */
  inline int get_rank() { return this->_rank; }

  /**
   * @brief Retrieves the number of GPUs to be used.
   * 
   * @return The number of GPUs specified for computation.
   */
  inline int get_gpu_count() { return this->_gpu_count; }

  /**
   * @brief Initializes the command line options.
   * 
   * Sets up the available command line options that the program accepts.
   */
  void Initialize();

  /**
   * @brief Validates the input file specified in the command line options.
   * 
   * Checks if the input file exists and is accessible.
   * 
   * @return `true` if the file is valid, `false` otherwise.
   */
  bool ValidateFile();

 private:
  po::options_description _options; ///< Describes the available command line options.
  std::string _input_path;           ///< Path to the input file.
  int _order;                        ///< Order (rank) of the tensor.
  int _rank;                         ///< Rank for the Tucker decomposition.
  int _gpu_count;                    ///< Number of GPUs to use for computation.

}; // class CommandLineOptions

/**
 * @brief Retrieves the input path specified in the command line options.
 * 
 * This function retrieves the input path specified in the command line options.
 * If no input path is specified, it returns a reference to an empty string.
 * 
 * @return A constant reference to the input path string.
 */
inline const std::string &CommandLineOptions::get_input_path() const {
  static const std::string empty_str;
  return (0 < this->_input_path.size() ? this->_input_path : empty_str);
}

}  // namespace gputucker
}  // namespace supertensor

#endif  // CMDLINE_OPTS_HPP_
