#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include "util.hpp"

namespace NeuralRun {
    namespace Quad {
        constexpr int INPUT_SIZE = 11;
        constexpr int OUTPUT_SIZE = 2;
        constexpr int A_LIMIT = 1;
        constexpr int B_LIMIT = 0;
        constexpr int C_LIMIT = 0;
        
        constexpr int DEFAULT_TRAIN_SIZE = 10000;
        constexpr int DEFAULT_TEST_SIZE = 2000;
        constexpr int DEFAULT_EPOCHS = 10;
        constexpr double DEFAULT_LEARNING_RATE = 0.001;
        constexpr int DEFAULT_BATCH_SIZE = 32;
        constexpr int DEFAULT_HIDDEN_SIZE = 1;
        
        void make_data(Eigen::MatrixXd& data, Eigen::MatrixXd& labels, int size) {
            RNG rng;
            for (int i = 0; i < size; ++i) {
                double a = rng.get_double(-A_LIMIT, A_LIMIT);
                double b = rng.get_double(-B_LIMIT, B_LIMIT);
                double c = rng.get_double(-C_LIMIT, C_LIMIT);
                for (int x = 0; x < INPUT_SIZE; ++x) {
                    data(x, i) = a * x * x + b * x + c;
                }
                labels(1, i) = a > 0;
                labels(0, i) = 1 - labels(1, i);
            }
        }
        
        void run(std::vector<std::string> arguments) {
            int argument_count = arguments.size();
            int train_size = argument_count > 0 ? std::stoi(arguments[0]) : DEFAULT_TRAIN_SIZE;
            int test_size = argument_count > 1 ? std::stoi(arguments[1]) : DEFAULT_TEST_SIZE;
            int epochs = argument_count > 2 ? std::stoi(arguments[2]) : DEFAULT_EPOCHS;
            double learning_rate = argument_count > 3 ? std::stod(arguments[3]) : DEFAULT_LEARNING_RATE;
            int batch_size = argument_count > 4 ? std::stoi(arguments[4]) : DEFAULT_BATCH_SIZE;
            int hidden_size = argument_count > 5 ? std::stoi(arguments[5]) : DEFAULT_HIDDEN_SIZE;
            std::cout << " Training set size: [" << train_size << "]\n"
                      << " Test set size: [" << test_size << "]\n"
                      << " Training epochs: [" << epochs << "]\n"
                      << " Learning rate: [" << learning_rate << "]\n"
                      << " Mini-batch size: [" << batch_size << "]\n"
                      << " Input size: [" << INPUT_SIZE << "]\n"
                      << " Hidden layer size: [" << hidden_size << "]\n"
                      << " Output size: [" << OUTPUT_SIZE << ']' << std::endl;
            
            std::cout << "\nGenerating data..." << std::endl;
            Eigen::MatrixXd train_data(INPUT_SIZE, train_size);
            Eigen::MatrixXd train_labels(OUTPUT_SIZE, train_size);
            Eigen::MatrixXd test_data(INPUT_SIZE, test_size);
            Eigen::MatrixXd test_labels(OUTPUT_SIZE, test_size);
            make_data(train_data, train_labels, train_size);
            make_data(test_data, test_labels, test_size);
            
            std::cout << "\nConstructing architecture..." << std::endl;
            MiniDNN::Network network;
            network.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(INPUT_SIZE, hidden_size));
            network.add_layer(new MiniDNN::FullyConnected<MiniDNN::Identity>(hidden_size, OUTPUT_SIZE));
            network.set_output(new MiniDNN::BinaryClassEntropy());
            MiniDNN::VerboseCallback callback;
            network.set_callback(callback);
            network.init();
            
            std::cout << "\nTraining..." << std::endl;
            MiniDNN::Adam optimiser(learning_rate);
            network.fit(optimiser, train_data, train_labels, batch_size, epochs);
            
            NeuralUtil::test(network, test_data, test_labels, test_size);
        }
    }
}