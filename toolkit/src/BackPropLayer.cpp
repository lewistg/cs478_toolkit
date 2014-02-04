#include <unistd.h>
#include "BackPropLayer.h"

BackPropLayer::BackPropLayer():
	_units(), 
	_prevLayer(NULL),
	_nextLayer(NULL)
{

}

BackPropLayer::BackPropLayer(size_t nUnits):
	_units(nUnits),
	_prevLayer(NULL),
	_nextLayer(NULL)
{
}

BackPropLayer::~BackPropLayer()
{

}

std::vector<double> BackPropLayer::trainOnExample(const std::vector<double>& input, const std::vector<double>& target)
{
	// calculate net outputs
	std::vector<double> layerOutputs;
	for(size_t i  = 0; i < _units.size(); i++)
	{
		layerOutputs.push_back(_units[i].getOutput(input));
	}

	std::vector<double> layerError;
	bool isInternalLayer = (_nextLayer != NULL);
	if(isInternalLayer)
	{
		// pass it onto the next layer
		std::vector<double> nextLayerError = _nextLayer->trainOnExample(layerOutputs, target);

		// calculate this layer's error
		for(size_t i  = 0; i < _units.size(); i++)
		{
			double errorCausedByUnit = 0.0;
			for(size_t j = 0; j < nextLayerError.size(); j++)
			{
				errorCausedByUnit += nextLayerError[i] * _units[i].getWeight(i);
			}	
			layerError.push_back(layerOutputs[i] * (1 - layerOutputs[i]));
			_units[i].updateWeights(errorCausedByUnit);
		}
	}
	else
	{
		for(size_t i  = 0; i < _units.size(); i++)
		{
			double error = layerOutputs[i] * (1 - layerOutputs[i]) * (input[i] - target[i]);
			layerError.push_back(error);
			_units[i].updateWeights(error);
		}
	}

    return layerError;
}

const BackPropUnit& BackPropLayer::operator[](size_t i)
{
	assert(i < _units.size());
	return _units[i];
}