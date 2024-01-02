#pragma once

#include <stdexcept>
#include <chrono>
#include <limits>
#include <random>
#include <ctime>
#include <MiniDNN.h>

/**
 * A namespace for time related functions.
 */
namespace Timer {
    // Constants for converting time bases.
    constexpr double HOURS_TO_MINUTES = 60;
    constexpr double MINUTES_TO_SECONDS = 60;
    
    /**
     * Returns the time in seconds elapsed since the epoch.
     */
    int current() noexcept {
        return std::time(NULL);
    }
    
    /**
     * Returns the time in seconds.
     * All times returned are relative to each other.
     */
    double time() noexcept {
        return static_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    /**
     * Halts all functionality in the thread for the
     *   specified amount of time (in seconds).
     */
    void wait(double seconds) noexcept {
        double now = time();
        while (time() < now + seconds);
    }

    /**
     * A constant expression function that converts time from
     *   hours, minutes, and seconds to just seconds.
     */
    constexpr double to_seconds(double hours = 0, double minutes = 0, double seconds = 0) noexcept {
        return HOURS_TO_MINUTES * MINUTES_TO_SECONDS * hours + MINUTES_TO_SECONDS * minutes + seconds;
    }
}

/**
 * A namespace for RNG functions.
 */
namespace Random {
    #define RANDOM_ASSERTION (min > max)
    constexpr const char* RANDOM_ERROR = "min can't be greater than max.";
    
    /**
     * A function that returns a random integer in the interval [min, max].
     * This function is a cross-platform replacement for std::uniform_int_distribution.
     */
    int get_int(std::mt19937& generator, int min, int max) {
        if (RANDOM_ASSERTION) {
            throw std::runtime_error(RANDOM_ERROR);
        }
        return min + generator() % static_cast<unsigned>(1 + max - min);
    }
    
    /**
     * A function that returns a random real number in the interval [min, max).
     * This function is a cross-platform replacement for std::uniform_real_distribution.
     */
    double get_real(std::mt19937& generator, double min, double max) {
        if (RANDOM_ASSERTION) {
            throw std::runtime_error(RANDOM_ERROR);
        }
        return min + (max - min) * generator() / (1 + static_cast<double>(std::numeric_limits<unsigned>::max()));
    }
    
    /**
     * A function that returns a random real number in the interval [min, max].
     */
    double get_double(std::mt19937& generator, double min, double max) {
        if (RANDOM_ASSERTION) {
            throw std::runtime_error(RANDOM_ERROR);
        }
        return min + (max - min) * generator() / static_cast<double>(std::numeric_limits<unsigned>::max());
    }
    
    #undef RANDOM_ASSERTION
}

/**
 * A class that encapsulates the functionality of the Random namespace with an internal generator.
 */
class RNG {
    public:
        /**
         * Seeds the instance with the current time.
         */
        RNG() noexcept: generator(Timer::current()) {}
        
        /**
         * Seeds the instance with the given value.
         */
        RNG(int seed) noexcept: generator(seed) {}
        
        /**
         * A method that returns a random integer in the interval [min, max].
         * This method is a cross-platform replacement for std::uniform_int_distribution.
         */
        int get_int(int min, int max) {
            return Random::get_int(generator, min, max);
        }
        
        /**
         * A method that returns a random real number in the interval [min, max).
         * This method is a cross-platform replacement for std::uniform_real_distribution.
         */
        double get_real(double min, double max) {
            return Random::get_real(generator, min, max);
        }
        
        /**
         * A method that returns a random real number in the interval [min, max].
         */
        double get_double(double min, double max) {
            return Random::get_double(generator, min, max);
        }
        
    private:
        std::mt19937 generator;
};

namespace NeuralUtil {
    class ProbabilisticCallback : public MiniDNN::Callback {
        public:
            ProbabilisticCallback(MiniDNN::Callback& callback, double probability) noexcept:
                callback(callback),
                probability(probability)
            {}
            
            void pre_training_batch(const MiniDNN::Network* net, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
                if (this->should_act()) {
                    this->set_callback_members();
                    this->callback.pre_training_batch(net, x, y);
                }
            }
            
            void pre_training_batch(const MiniDNN::Network* net, const Eigen::MatrixXd& x, const Eigen::RowVectorXi& y) {
                if (this->should_act()) {
                    this->set_callback_members();
                    this->callback.pre_training_batch(net, x, y);
                }
            }
            
            void post_training_batch(const MiniDNN::Network* net, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
                if (this->should_act()) {
                    this->set_callback_members();
                    this->callback.post_training_batch(net, x, y);
                }
            }
            
            void post_training_batch(const MiniDNN::Network* net, const Eigen::MatrixXd& x, const Eigen::RowVectorXi& y) {
                if (this->should_act()) {
                    this->set_callback_members();
                    this->callback.post_training_batch(net, x, y);
                }
            }
        
        private:
            bool should_act() noexcept {
                return this->rng.get_real(0, 1) < this->probability;
            }
            
            void set_callback_members() noexcept {
                this->callback.m_nbatch = this->m_nbatch;
                this->callback.m_batch_id = this->m_batch_id;
                this->callback.m_nepoch = this->m_nepoch;
                this->callback.m_epoch_id = this->m_epoch_id;
            }
            
            RNG rng;
            MiniDNN::Callback& callback;
            double probability;
    };
    
    int get_max_column_index(const Eigen::MatrixXd& matrix, int row_index, int columns) noexcept {
        int max_index = 0;
        double max = matrix(0, row_index);
        for (int i = 1; i < columns; ++i) {
            double value = matrix(i, row_index);
            if (max < value) {
                max = value;
                max_index = i;
            }
        }
        return max_index;
    }
    
    void test_classifier(MiniDNN::Network& network,
                         const Eigen::MatrixXd& test_data,
                         const Eigen::MatrixXd& test_labels,
                         int test_size,
                         int input_size,
                         int output_size,
                         double log_probability) noexcept {
        std::cout << "\nTesting..." << std::endl;
        std::cout << "Outputting " << 100 * log_probability << "% of test cases." << std::endl;
        RNG rng;
        Eigen::MatrixXd prediction(network.predict(test_data));
        double correct = 0;
        
        for (int i = 0; i < test_size; ++i) {
            int predicted = get_max_column_index(prediction, i, output_size);
            int expected = get_max_column_index(test_labels, i, output_size);
            if (predicted == expected) {
                ++correct;
            }
            
            if (rng.get_real(0, 1) < log_probability) {
                std::cout << "( ";
                for (int j = 0; j < input_size; ++j) {
                    std::cout << test_data(j, i) << ' ';
                }
                std::cout << ") -> " << expected << " (Predicted " << predicted << ": [ ";
                for (int j = 0; j < output_size; ++j) {
                    std::cout << prediction(j, i) << ' ';
                }
                std::cout << "])" << std::endl;
            }
        }
        
        std::cout << "Accuracy: " << correct / test_size << std::endl;
    }
}
