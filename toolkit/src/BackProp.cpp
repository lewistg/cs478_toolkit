#include <vector>
#include "BackProp.h"
#include "data_utils.h"

BackProp::BackProp(Rand& rand, bool loggingOn):_loggingOn(loggingOn)
{
}

void BackProp::train(Matrix& features, Matrix& labels)
{
	// create the layers
    std::vector<size_t> layerConfig;
    layerConfig.push_back(features.cols());
    layerConfig.push_back(features.cols());
    layerConfig.push_back(features.cols());

	size_t outputUnits = 0;
	for(size_t i = 0; i < labels.cols(); i++)
		outputUnits += (labels.valueCount(i) > 0 ? labels.valueCount(i) : 1); 
	layerConfig.push_back(outputUnits);

	createLayers(layerConfig);

	setupTest(*this);

	if(_loggingOn)
	{
		std::cout << "Network created..." << std::endl;
		for(size_t i = 0; i < _layers.size(); i++)
		{
			std::cout << _layers[i].toString() << std::endl;
		}
	}

	assert(_layers.size() > 0);
	assert(features.cols() == _layers[0].getNumUnits());

	for(size_t i = 0; i < features.rows(); i++)
	{
		if(_loggingOn)
		{
			std::cout << "Training example input vector: " << vectorToString(features.row(i)) << std::endl;
			std::cout << "Target label: " << vectorToString(labels.row(i)) << std::endl;
		}

		_layers[0].trainOnExample(features.row(i), labels.row(i));
	}
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
		size_t layerIndex = i - 1;
        _layers.push_back(BackPropLayer(layerConfig[i], layerIndex, _loggingOn));
		if(layerIndex == 0)
		{
			size_t inputLayerConfig = 0;
			_layers[layerIndex].setNumInputs(layerConfig[inputLayerConfig]);
		}
		else
		{
			_layers[layerIndex].setPrevLayer(&_layers[layerIndex - 1]);
			_layers[layerIndex - 1].setNextLayer(&_layers[layerIndex]);
		}
	}
}

void setupTest(BackProp& backProp)
{
	assert(backProp._layers.size() == 3);

	// setup the weights
	assert(backProp._layers[0].getNumUnits() == 2);
	assert(backProp._layers[1].getNumUnits() == 2);
	assert(backProp._layers[2].getNumUnits() == 2);

	std::vector<double> w;
	w.push_back(0.2);
	w.push_back(-0.1);
	w.push_back(0.1);

	backProp._layers[0].getUnit(0).setWeights(w);

	w[0] = 0.3;
	w[1] = -0.3;
	w[2] = -0.2;
	backProp._layers[0].getUnit(1).setWeights(w);

	w[0] = -0.2;
	w[1] = -0.3;
	w[2] = 0.1;
	backProp._layers[1].getUnit(0).setWeights(w);

	w[0] = -0.1;
	w[1] = 0.3;
	w[2] = 0.2;
	backProp._layers[1].getUnit(1).setWeights(w);

	w[0] = -0.1;
	w[1] = 0.3;
	w[2] = 0.2;
	backProp._layers[2].getUnit(0).setWeights(w);

	w[0] = -0.2;
	w[1] = -0.3;
	w[2] = 0.1;
	backProp._layers[2].getUnit(1).setWeights(w);
}