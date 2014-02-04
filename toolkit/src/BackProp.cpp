#include <vector>
#include "BackProp.h"

BackProp::BackProp(Rand& rand, const std::vector<size_t>& layerConfig)
{
	// construct the network
	for(size_t i = 0; i < layerConfig.size(); i++)
	{
	}
}

void BackProp::train(Matrix& features, Matrix& labels)
{
	// train each of the input 
}

void BackProp::predict(const std::vector<double>& features, std::vector<double>& labels)
{

}