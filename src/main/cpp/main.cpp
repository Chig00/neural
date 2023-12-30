#include <iostream>
#include <MiniDNN.h>
#include "util.hpp"

int main() {
	std::cout << "\nGenerating data...\n";
	RNG rng;
	Eigen::MatrixXd train_data(100, 2);
	Eigen::MatrixXd train_labels(100, 1);
	Eigen::MatrixXd test_data(10, 2);
	Eigen::MatrixXd test_labels(10, 1);
	
	for (int i = 0; i < 100; ++i) {
		train_data(i, 0) = rng.get_double(0, 1);
		train_data(i, 1) = rng.get_double(0, 1);
		train_labels(i, 0) = train_data(i, 0) > 0.5 && train_data(i, 1) > 0.5 ? 1 : -1;
	}
	for (int i = 0; i < 10; ++i) {
		test_data(i, 0) = rng.get_double(0, 1);
		test_data(i, 1) = rng.get_double(0, 1);
		test_labels(i, 0) = test_data(i, 0) > 0.5 && test_data(i, 1) > 0.5 ? 1 : -1;
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
	return 0;
}
