#include <iostream>
#include <cmath>
#include <MiniDNN.h>
#include "util.hpp"

constexpr int INPUT_SIZE = 2;
constexpr int HIDDEN_SIZE = 4;
constexpr int OUTPUT_SIZE = 1;
constexpr int TRAIN_SIZE = 100;
constexpr int TEST_SIZE = 10;
constexpr int BATCH_SIZE = 10;
constexpr int EPOCHS = 10;

int main() {
	std::cout << "\nGenerating data..." << std::endl;
	RNG rng;
	Eigen::MatrixXd train_data(INPUT_SIZE, TRAIN_SIZE);
	Eigen::MatrixXd train_labels(OUTPUT_SIZE, TRAIN_SIZE);
	Eigen::MatrixXd test_data(INPUT_SIZE, TEST_SIZE);
	Eigen::MatrixXd test_labels(OUTPUT_SIZE, TEST_SIZE);
	
	for (int i = 0; i < TRAIN_SIZE; ++i) {
		train_data(0, i) = rng.get_double(0, 1);
		train_data(1, i) = rng.get_double(0, 1);
		train_labels(0, i) = train_data(0, i) > 0.5 && train_data(1, i) > 0.5;
	}
	for (int i = 0; i < TEST_SIZE; ++i) {
		test_data(0, i) = rng.get_double(0, 1);
		test_data(1, i) = rng.get_double(0, 1);
		test_labels(0, i) = test_data(0, i) > 0.5 && test_data(1, i) > 0.5;
	}
	
	std::cout << "\nConstructing architecture..." << std::endl;
	MiniDNN::Network network;
	network.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(INPUT_SIZE, HIDDEN_SIZE));
	network.add_layer(new MiniDNN::FullyConnected<MiniDNN::Identity>(HIDDEN_SIZE, OUTPUT_SIZE));
	network.set_output(new MiniDNN::BinaryClassEntropy());
	MiniDNN::VerboseCallback callback;
	network.set_callback(callback);
	network.init();
	
	std::cout << "\nTraining..." << std::endl;
	MiniDNN::AdaGrad optimiser;
	network.fit(optimiser, train_data, train_labels, BATCH_SIZE, EPOCHS);
	
	std::cout << "\nTesting..." << std::endl;
	Eigen::MatrixXd prediction(network.predict(test_data));
	double correct = 0;
	for (int i = 0; i < TEST_SIZE; ++i) {
		double predicted = prediction(0, i);
		double expected = test_labels(0, i);
		if (std::round(predicted) == expected) {
			++correct;
		}
		
		std::cout << '(' << test_data(0, i) << ", " << test_data(1, i) << ") -> " << expected
				  << " (predicted " << predicted << ')' << std::endl;
	}
	std::cout << "\nAccuracy: " << correct / TEST_SIZE << std::endl;
	return 0;
}
