#include <sstream>
#include "small.hpp"
#include "quad.hpp"
#include "ranges.hpp"

constexpr int RUN_MODE_INDEX = 1;
constexpr int RUN_ARGUMENTS_INDEX = 2;
constexpr char RUN_ARGUMENT_SPLITTER = ' ';

int main(int argc, char** argv) {
    if (argc <= RUN_MODE_INDEX) {
        std::ostringstream stream;
        stream << "Invalid number of arguments. Expected at least: [" << RUN_MODE_INDEX
               << "]. Received: [" << argc - 1 << "].";
        throw std::runtime_error(stream.str());
    }
    
    std::string run_mode(argv[RUN_MODE_INDEX]);
    std::cout << "\nUsing run mode: [" << run_mode << "]." << std::endl;
    std::vector<std::string> run_arguments;
    if (argc > RUN_ARGUMENTS_INDEX) {
        std::istringstream run_arguments_stream(argv[RUN_ARGUMENTS_INDEX]);
        std::cout << "Using run arguments: [" << run_arguments_stream.str() << "]." << std::endl;
        for (std::string argument; std::getline(run_arguments_stream, argument, RUN_ARGUMENT_SPLITTER);) {
            run_arguments.push_back(argument);
        }
    } else {
        std::cout << "Using default run arguments." << std::endl;
    }
    
    if (run_mode == "small") {
        NeuralRun::Small::run(run_arguments);
    } else if (run_mode == "quad") {
        NeuralRun::Quad::run(run_arguments);
    } else if (run_mode == "ranges") {
        NeuralRun::Ranges::run(run_arguments);
    } else {
        std::ostringstream stream;
        stream << "Unknown run mode: [" << run_mode << "].";
        throw std::runtime_error(stream.str());
    }
}
