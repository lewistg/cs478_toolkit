#include <vector>
#include "BackProp.h"

BackProp::BackProp(Rand& rand, const std::vector<size_t>& layerConfig)
{
	// construct the network
	for(size_t i = 0; i < layerConfig.size(); i++)
	{
		assert(layerConfig[i] > 0);
        _layers.push_back(BackPropLayer(layerConfig[i]));
		if(i > 0)
		{
			_layers[i].setPrevLayer(&_layers[i - 1]);
			_layers[i - 1].setNextLayer(&_layers[i]);
		}
	}
}

void BackProp::train(Matrix& features, Matrix& labels)
{
	assert(_layers.size() > 0);
	assert(features.cols() == _layers[0].getNumUnits());

	for(size_t i = 0; i < features.rows(); i++)
		_layers[0].trainOnExample(features.row(i), labels.row(i));
}

void BackProp::predict(const std::vector<double>& features, std::vector<double>& labels)
{

}