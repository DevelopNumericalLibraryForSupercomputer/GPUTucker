#include "gputucker/cmdline_opts.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

#include "gputucker/helper.hpp"
namespace supertensor {
namespace gputucker {

CommandLineOptions::CommandLineOptions()
    : input_indices_path_(""),
      input_values_path_(""),
      order_(3),
      rank_(10),
      gpu_count_(1),
      gpts_(1) {
  initialize();
}

CommandLineOptions::~CommandLineOptions() {}

void CommandLineOptions::initialize() {
  po::options_description options("Program Options");
  options.add_options()("help,h", "Display help menu.")(
      "indices,i", po::value<std::string>(&this->input_indices_path_),
      "Input tensor indices path")(
      "values,v", po::value<std::string>(&this->input_values_path_),
      "Input tensor values path")("order,o", po::value<int>(&this->order_),
                                  "Order")(
      "rank,r", po::value<int>(&this->rank_)->default_value(10), "Rank")(
      "gpus,g", po::value<int>(&this->gpu_count_)->default_value(1),
      "The number of GPUs")("gpts,q",
                            po::value<uint64_t>(&this->gpts_)->default_value(1),
                            "Quantum maximum index value");

  this->options_.add(options);
}

CommandLineOptions::ReturnStatus CommandLineOptions::parse(int argc,
                                                           char *argv[]) {
  ReturnStatus ret = OPTS_SUCCESS;

  po::variables_map var_map;
  char file_name[5000];

  try {
    po::store(po::parse_command_line(argc, argv, this->options_), var_map);
    po::notify(var_map);

    // Help option
    if (var_map.count("help")) {
      std::cout << this->options_ << std::endl;
      return OPTS_HELP;
    }

    // Enforce an input file argument every time
    if (!(0 < var_map.count("indices"))) {
      std::cout << CYN "[ERROR] Input file must be specified!!!" RESET
                << std::endl;
      std::cout << this->options_ << std::endl;
      return OPTS_FAILURE;
    } else {
      // Strip whitespaces from front/back of filename string
      boost::algorithm::trim(this->input_indices_path_);

      // Resolve the filename to be fully-qualified
      realpath(this->input_indices_path_.c_str(), file_name);
      this->input_indices_path_ = file_name;

      ret = validate_files() ? OPTS_SUCCESS : OPTS_FAILURE;
    }
    // Enforce an input file argument every time
    if (!(0 < var_map.count("values"))) {
      std::cout << CYN "[ERROR] Input file must be specified!!!" RESET
                << std::endl;
      std::cout << this->options_ << std::endl;
      return OPTS_FAILURE;
    } else {
      // Strip whitespaces from front/back of filename string
      boost::algorithm::trim(this->input_values_path_);

      // Resolve the filename to be fully-qualified
      realpath(this->input_values_path_.c_str(), file_name);
      this->input_values_path_ = file_name;

      ret = validate_files() ? OPTS_SUCCESS : OPTS_FAILURE;
    }

    if (!(0 < var_map.count("order"))) {
      std::cout << "[ERROR] Tensor order must be specified!!!" << std::endl;
      std::cout << this->options_ << std::endl;
      return OPTS_FAILURE;
    }

    if (!(0 < var_map.count("gpts"))) {
      std::cout << "[ERROR] quantum maximum index value must be specified!!!"
                << std::endl;
      std::cout << this->options_ << std::endl;
      return OPTS_FAILURE;
    }

    // We can check if a rank is defaulted
    if (!var_map["rank"].defaulted()) {
      std::cout << "[WARNING] Default value for User-Value overwritten to "
                << this->rank_ << std::endl;
    }
  } catch (std::exception &e) {
    std::cout << "[ERROR] Parsing error : " << e.what() << std::endl;
    ret = OPTS_FAILURE;
  } catch (...) {
    std::cout << "[ERROR] Parsing error (unknown type)." << std::endl;
    ret = OPTS_FAILURE;
  }

  return ret;
}

bool CommandLineOptions::validate_files() {
  if (!boost::filesystem::is_regular_file(this->input_indices_path_)) {
    std::cout << "ERROR - Input file provided does not exist ["
              << this->input_indices_path_ << "]" << std::endl;
    return false;
  }

  if (!boost::filesystem::is_regular_file(this->input_values_path_)) {
    std::cout << "ERROR - Input file provided does not exist ["
              << this->input_values_path_ << "]" << std::endl;
    return false;
  }
  return true;
}

}  // namespace gputucker
}  // namespace supertensor