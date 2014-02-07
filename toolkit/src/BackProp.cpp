#include <vector>
#include "BackProp.h"

BackProp::BackProp(Rand& rand, bool loggingOn):_loggingOn(loggingOn)
{
}

void BackProp::train(Matrix& features, Matrix& labels)
{
	// create the layers
    std::vector<size_t> layerConfig;
    layerConfig.push_back(features.cols());
    layerConfig.push_back(features.cols());

	size_t outputUnits = 0;
	for(size_t i = 0; i < labels.cols(); i++)
		outputUnits += (labels.valueCount(i) > 0 ? labels.valueCount(i) : 1); 
	layerConfig.push_back(outputUnits);

	createLayers(layerConfig);

	assert(_layers.size() > 0);
	assert(features.cols() == _layers[0].getNumUnits());

	for(size_t i = 0; i < features.rows(); i++)
		_layers[0].trainOnExample(features.row(i), labels.row(i));
}

void BackProp::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	assert(_layers.size() > 0);
	assert(features.size() == _layers[0].getNumUnits());
	std::vector<double> finalLayerOutput;
	//_layers[0].predict(features, finalLayerOutput);
}

void BackProp::createLayers(const std::vector<size_t>& layerConfig)
{
    _layers.clear();
	for(size_t i = 1; i < layerConfig.size(); i++)
	{
		assert(layerConfig[i] > 0);
        _layers.push_back(BackPropLayer(layerConfig[i], _loggingOn));
		size_t layerIndex = i - 1;
		size_t inputLayerConfig = 0;
		if(layerIndex == 0)
		{
			_layers[layerIndex].setNumInputs(layerConfig[inputLayerConfig]);
		}
		else
		{
			_layers[layerIndex].setPrevLayer(&_layers[layerIndex - 1]);
			_layers[layerIndex - 1].setNextLayer(&_layers[layerIndex]);
		}
	}
}