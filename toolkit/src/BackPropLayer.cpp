#include <unistd.h>
#include <iostream>
#include "BackPropLayer.h"

BackPropLayer::BackPropLayer(bool loggingOn = false):
	_units(), 
	_prevLayer(NULL),
	_nextLayer(NULL),
	_layerId(0),
	_loggingOn(loggingOn)
{

}

BackPropLayer::BackPropLayer(size_t nUnits, bool loggingOn = false):
	_units(nUnits),
	_prevLayer(NULL),
	_nextLayer(NULL),
	_layerId(0),
	_loggingOn(loggingOn)
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

	size_t nInputs = _prevLayer->getNumUnits();
	for(size_t i = 0; i < _units.size(); i++)
		_units[i].setNumInputs(nInputs);
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
			double error = layerOutputs[i] * (1 - layerOutputs[i]) * errorCausedByUnit;
			layerError.push_back(error);

			_units[i].updateWeights(error);
		}

		if(_loggingOn)
		{
			logLayerError(layerError);
			logUnitWeights();
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
		labels.clear();
		for(size_t j = 0; j < layerOutputs.size(); j++)
			labels.push_back(layerOutputs[j]);
	}
}

void BackPropLayer::setLayerId(size_t layerId)
{
	_layerId = layerId;
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

void BackPropLayer::setNumInputs(size_t nInputs) 
{
	for(size_t i = 0; i < _units.size(); i++)
		_units[i].setNumInputs(nInputs);
}

std::string BackPropLayer::toString()
{
	std::stringstream ss;
	ss << "(layer id =  " << _layerId << ", " << std::endl;
	for(size_t i = 0; i < _units.size(); i++)
		ss << "\t" << _units[i].toString() << std::endl;
	ss << ")";

	return ss.str();
}

void BackPropLayer::logLayerError(const std::vector<double>& layerError)
{
	std::cout << "Layer " << _layerId << " error:";
	for(size_t i = 0; i < layerError.size(); i++)
	{
		std::cout << layerError[i];
		if(i < layerError.size() - 1)
			std::cout << ", ";
	}
	std::cout << std::endl;
}

void BackPropLayer::logUnitWeights()
{
	std::cout << "Layer " << _layerId << " unit weights:";
	for(size_t i = 0; i < _units.size(); i++)
		std::cout << _units[i].toString() << std::endl;
}