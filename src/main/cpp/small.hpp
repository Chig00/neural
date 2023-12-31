#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <MiniDNN.h>
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
            RNG rng;
            Eigen::MatrixXd train_data(INPUT_SIZE, train_size);
            Eigen::MatrixXd train_labels(OUTPUT_SIZE, train_size);
            Eigen::MatrixXd test_data(INPUT_SIZE, test_size);
            Eigen::MatrixXd test_labels(OUTPUT_SIZE, test_size);
            
            for (int i = 0; i < train_size; ++i) {
                train_data(0, i) = rng.get_double(0, 1);
                train_data(1, i) = rng.get_double(0, 1);
                train_labels(1, i) = train_data(0, i) > 0.5 && train_data(1, i) > 0.5;
                train_labels(0, i) = 1 - train_labels(1, i);
            }
            for (int i = 0; i < test_size; ++i) {
                test_data(0, i) = rng.get_double(0, 1);
                test_data(1, i) = rng.get_double(0, 1);
                test_labels(1, i) = test_data(0, i) > 0.5 && test_data(1, i) > 0.5;
                test_labels(0, i) = 1 - test_labels(1, i);
            }
            
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
            
            std::cout << "\nTesting..." << std::endl;
            Eigen::MatrixXd prediction(network.predict(test_data));
            double correct = 0;
            for (int i = 0; i < test_size; ++i) {
                double predicted_0 = prediction(0, i);
                double predicted_1 = prediction(1, i);
                int predicted = predicted_1 > predicted_0;
                int expected = test_labels(1, i);
                if (predicted == expected) {
                    ++correct;
                }
                
                std::cout << '(' << test_data(0, i) << ", " << test_data(1, i) << ") -> " << expected
                          << " (Predicted " << predicted << ": [" << predicted_0 << ", " << predicted_1 << "])"
                          << std::endl;
            }
            std::cout << "\nAccuracy: " << correct / test_size << std::endl;
        }
    }
}
