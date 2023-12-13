#include "gputucker/cmdline_opts.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

#include "gputucker/helper.hpp"
namespace supertensor {
namespace gputucker {
CommandLineOptions::CommandLineOptions()
    : _input_path(""), _order(3), _rank(10), _gpu_count(1) {
  initialize();
}

CommandLineOptions::~CommandLineOptions() {}

void CommandLineOptions::initialize() {
  po::options_description options("Program Options");
  options.add_options()("help,h", "Display help menu.")(
      "input,i", po::value<std::string>(&this->_input_path),
      "Input tensor path")("order,o", po::value<int>(&this->_order), "Order")(
      "rank,r", po::value<int>(&this->_rank)->default_value(10), "Rank")(
      "gpus,g", po::value<int>(&this->_gpu_count)->default_value(1),
      "The number of GPUs");

  this->_options.add(options);
}

CommandLineOptions::ReturnStatus CommandLineOptions::parse(int argc,
                                                           char *argv[]) {
  ReturnStatus ret = OPTS_SUCCESS;

  po::variables_map var_map;
  char file_name[5000];

  try {
    po::store(po::parse_command_line(argc, argv, this->_options), var_map);
    po::notify(var_map);

    // Help option
    if (var_map.count("help")) {
      std::cout << this->_options << std::endl;
      return OPTS_HELP;
    }

    // Enforce an input file argument every time
    if (!(0 < var_map.count("input"))) {
      std::cout << CYN "[ERROR] Input file must be specified!!!" RESET
                << std::endl;
      std::cout << this->_options << std::endl;
      return OPTS_FAILURE;
    } else {
      // Strip whitespaces from front/back of filename string
      boost::algorithm::trim(this->_input_path);

      // Resolve the filename to be fully-qualified
      if (realpath(this->_input_path.c_str(), file_name) == NULL) {
        std::cout << CYN "[ERROR] Input file provided does not exist ["
                  << this->_input_path << "]" << std::endl;
        return OPTS_FAILURE;
      }
      this->_input_path = file_name;

      ret = validate_files() ? OPTS_SUCCESS : OPTS_FAILURE;
    }
    if (!(0 < var_map.count("order"))) {
      std::cout << CYN "[ERROR] Tensor order must be specified!!!" << std::endl;
      std::cout << this->_options << std::endl;
      return OPTS_FAILURE;
    }

    // We can check if a rank is defaulted
    if (!var_map["rank"].defaulted()) {
      std::cout << "[WARNING] Default value for User-Value overwritten to "
                << this->_rank << std::endl;
    }

    if (!var_map["gpus"].defaulted()) {
      std::cout << "[WARNING] Default value for GPU count overwritten to "
                << this->_gpu_count << std::endl;
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
  if (!boost::filesystem::is_regular_file(this->_input_path)) {
    std::cout << "ERROR - Input file provided does not exist ["
              << this->_input_path << "]" << std::endl;
    return false;
  }
  return true;
}

}  // namespace gputucker
}  // namespace supertensor