#include <iostream>

#undef NOMINMAX
#pragma GCC diagnostic ignored "-Wdeprecated"
#include "../../../lib/opennn/opennn/opennn.h"
#pragma GCC diagnostic pop

int main() {
	// opennn::Index inputs = 2;
	// opennn::Index hidden = 4;
	// opennn::Index outputs = 2;
	// opennn::Tensor<Index, 1> architecture(3);
	// architecture(0) = inputs;
	// architecture(1) = first_hidden;
	// architecture(2) = outputs;
	// opennn::NeuralNetwork network(opennn::NeuralNetwork::ProjectType::Approximation, architecture);
	// std::cout << network.get_layers_neurons_numbers();
	opennn::NeuralNetwork network;
	return 0;
}
