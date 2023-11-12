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
  ReturnStatus parse(int argc, char *argv[]);
  const std::string &get_input_indices_path() const;
  const std::string &get_input_values_path() const;
  inline int get_order() { return this->order_; }
  inline int get_rank() { return this->rank_; }
  inline int get_gpu_count() { return this->gpu_count_; }
  inline uint64_t get_gpts() { return this->gpts_; }

 protected:
  void initialize();
  bool validate_files();

 private:
  CommandLineOptions(const CommandLineOptions &rhs);
  CommandLineOptions &operator=(const CommandLineOptions &rhs);

  po::options_description options_;
  std::string input_indices_path_;
  std::string input_values_path_;
  int order_;
  int rank_;
  int gpu_count_;
  uint64_t gpts_;  // quantum maximun index value
};

inline const std::string &CommandLineOptions::get_input_indices_path() const {
  static const std::string empty_str;
  return (0 < this->input_indices_path_.size() ? this->input_indices_path_
                                               : empty_str);
}

inline const std::string &CommandLineOptions::get_input_values_path() const {
  static const std::string empty_str;
  return (0 < this->input_values_path_.size() ? this->input_values_path_
                                              : empty_str);
}

}  // namespace gputucker
}  // namespace supertensor

#endif  // CMDLINE_OPTS_HPP_
