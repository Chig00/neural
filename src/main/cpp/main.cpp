#include <iostream>
#include <MiniDNN.h>
#include "util.hpp"

int main() {
	std::cout << "\nGenerating data...\n";
	RNG rng;
	Eigen::MatrixXd train_data(2, 100);
	Eigen::MatrixXd train_labels(2, 100);
	Eigen::MatrixXd test_data(2, 10);
	Eigen::MatrixXd test_labels(2, 10);
	
	for (int i = 0; i < 100; ++i) {
		train_data(0, i) = rng.get_double(0, 1);
		train_data(1, i) = rng.get_double(0, 1);
		train_labels(0, i) = train_data(0, i) > 0.5 && train_data(1, i) > 0.5;
		train_labels(1, i) = 1 - train_labels(0, i);
	}
	for (int i = 0; i < 10; ++i) {
		test_data(0, i) = rng.get_double(0, 1);
		test_data(1, i) = rng.get_double(0, 1);
		test_labels(0, i) = test_data(0, i) > 0.5 && test_data(1, i) > 0.5;
		test_labels(1, i) = 1 - test_labels(0, i);
	}
	
	std::cout << "\nConstructing architecture...\n";
	MiniDNN::Network network;
	network.add_layer(new MiniDNN::FullyConnected<MiniDNN::ReLU>(2, 4));
	network.add_layer(new MiniDNN::FullyConnected<MiniDNN::Identity>(4, 2));
	network.set_output(new MiniDNN::BinaryClassEntropy());
	MiniDNN::VerboseCallback callback;
	network.set_callback(callback);
	network.init();
	
	std::cout << "\nTraining...\n";
	MiniDNN::AdaGrad optimiser;
	network.fit(optimiser, train_data, train_labels, 100, 10);
	
	std::cout << "\nTesting...\n";
	Eigen::MatrixXd prediction(network.predict(test_data));
	for (int i = 0; i < 10; ++i) {
		std::cout << '(' << test_data(0, i) << ", " << test_data(1, i) << ") -> " << train_labels(0, i)
				  << " (predicted " << prediction(0, i) << ")\n";
	}
	return 0;
}
