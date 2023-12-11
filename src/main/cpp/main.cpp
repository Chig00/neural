#include <iostream>
#include <MiniDNN.h>

int main() {
	MiniDNN::Network network;
	std::cout << network.num_layers() << std::endl;
	return 0;
}
