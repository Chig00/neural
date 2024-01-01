#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include "util.hpp"

namespace NeuralRun {
    namespace Small {
        constexpr int INPUT_SIZE = 2;
        constexpr int OUTPUT_SIZE = 2;
        
        constexpr int DEFAULT_TRAIN_SIZE = 10000;
        constexpr int DEFAULT_TEST_SIZE = 2000;
        constexpr int DEFAULT_EPOCHS = 10;
        constexpr double DEFAULT_LEARNING_RATE = 0.001;
        constexpr int DEFAULT_BATCH_SIZE = 32;
        constexpr int DEFAULT_HIDDEN_SIZE = 3;
        constexpr double DEFAULT_TEST_OUTPUT_CHANCE = 0.01;
        
        void make_data(Eigen::MatrixXd& data, Eigen::MatrixXd& labels, int size) {
            RNG rng;
            for (int i = 0; i < size; ++i) {
                data(0, i) = rng.get_double(0, 1);
                data(1, i) = rng.get_double(0, 1);
                labels(1, i) = data(0, i) > 0.5 && data(1, i) > 0.5;
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
            double test_output_chance = argument_count > 6 ? std::stod(arguments[6]) : DEFAULT_TEST_OUTPUT_CHANCE;
            std::cout << " Training set size: [" << train_size << "]\n"
                      << " Test set size: [" << test_size << "]\n"
                      << " Training epochs: [" << epochs << "]\n"
                      << " Learning rate: [" << learning_rate << "]\n"
                      << " Mini-batch size: [" << batch_size << "]\n"
                      << " Input size: [" << INPUT_SIZE << "]\n"
                      << " Hidden layer size: [" << hidden_size << "]\n"
                      << " Output size: [" << OUTPUT_SIZE << "]\n"
                      << " Test Output Chance: [" << test_output_chance << ']' << std::endl;
            
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
            
            NeuralUtil::test_binary_classifier(network, test_data, test_labels, test_size, INPUT_SIZE, test_output_chance);
        }
    }
}
