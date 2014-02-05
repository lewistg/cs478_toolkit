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

void BackPropLayer::setNextLayer(BackPropLayer* nextLayer)
{
	_nextLayer = nextLayer;
}
	
void BackPropLayer::setPrevLayer(BackPropLayer* prevLayer)
{
	_prevLayer = prevLayer;
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

void BackPropLayer::predict(const std::vector<double>& input, std::vector<double>& labels) const
{
	std::vector<double> layerOutputs;
	for(size_t i  = 0; i < _units.size(); i++)
	{
		layerOutputs.push_back(_units[i].getOutput(input));
	}

	bool isOutputLayer = (_nextLayer == NULL);
	if(isOutputLayer)
	{
		assert(layerOutputs.size() == labels.size());
		for(size_t j = 0; j < layerOutputs.size(); j++)
			labels[j] = layerOutputs[j];
	}
}

const BackPropUnit& BackPropLayer::operator[](size_t i)
{
	assert(i < _units.size());
	return _units[i];
}

size_t BackPropLayer::getNumUnits() const
{
	return _units.size();
}