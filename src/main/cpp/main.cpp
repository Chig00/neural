#include <iostream>

#undef NOMINMAX
#pragma GCC diagnostic ignored "-Wdeprecated"
#include "../../../lib/opennn/opennn/opennn.h"
#pragma GCC diagnostic pop

int main() {
	opennn::Tensor<int, 1> tensor(5);
	for (int i = 0; i < 5; ++i) {
		tensor(i) = i;
	}
	std::cout << tensor;
	return 0;
}
