#ifndef CMDLINE_OPTS_HPP_
#define CMDLINE_OPTS_HPP_

#include <boost/program_options.hpp>
#include <cstdint>
#include <string>

namespace po = boost::program_options;

namespace supertensor {
namespace gputucker {

class CommandLineOptions {
 public:
  enum ReturnStatus { OPTS_SUCCESS, OPTS_HELP, OPTS_FAILURE };

  CommandLineOptions();
  ~CommandLineOptions();

  ReturnStatus        Parse(int argc, char *argv[]);
  const std::string&  get_input_path() const;
  inline int get_order()      { return this->_order; }
  inline int get_rank()       { return this->_rank; }
  inline int get_gpu_count()  { return this->_gpu_count; }

  void Initialize();
  bool ValidateFile();

 private:

  po::options_description _options;
  std::string _input_path;
  int _order;
  int _rank;
  int _gpu_count;
}; // class CommandLineOptions

inline const std::string &CommandLineOptions::get_input_path() const {
  static const std::string empty_str;
  return (0 < this->_input_path.size() ? this->_input_path : empty_str);
}

}  // namespace gputucker
}  // namespace supertensor

#endif  // CMDLINE_OPTS_HPP_
