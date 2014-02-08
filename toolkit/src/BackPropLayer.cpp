#include <unistd.h>
#include <iostream>
#include "BackPropLayer.h"
#include "data_utils.h"

BackPropLayer::BackPropLayer(bool loggingOn):
	_units(), 
	_prevLayer(NULL),
	_nextLayer(NULL),
	_layerId(0),
	_loggingOn(loggingOn)
{

}

BackPropLayer::BackPropLayer(size_t nUnits, size_t layerId, bool loggingOn):
	_units(nUnits, BackPropUnit(loggingOn)),
	_prevLayer(NULL),
	_nextLayer(NULL),
	_layerId(layerId),
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

BackPropLayer* BackPropLayer::getNextLayer() const
{
	return _nextLayer;
}
	
void BackPropLayer::setPrevLayer(BackPropLayer* prevLayer)
{
	_prevLayer = prevLayer;

	size_t nInputs = _prevLayer->getNumUnits();
	for(size_t i = 0; i < _units.size(); i++)
		_units[i].setNumInputs(nInputs);
}

BackPropLayer* BackPropLayer::getPrevLayer() const
{
	return _prevLayer;
}

std::vector<double> BackPropLayer::trainOnExample(const std::vector<double>& input, const std::vector<double>& target)
{
	if(_loggingOn)
	{
		std::cout << "Training layer: " << std::endl; 
		std::cout << toString() << std::endl;
		if(_nextLayer != NULL)
			std::cout << "Next layer: " << _nextLayer->toString() << std::endl;
		std::cout << "Layer input: " << vectorToString(input) << std::endl;
	}

	// calculate net outputs
	std::vector<double> layerOutputs;
	for(size_t i  = 0; i < _units.size(); i++)
	{
		layerOutputs.push_back(_units[i].getOutput(input));
	}

	if(_loggingOn)
	{
		std::cout << "Layer outputs: " << vectorToString(layerOutputs) << std::endl;
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
				errorCausedByUnit += nextLayerError[i] * _nextLayer->getUnit(j).getWeight(i);
			}	
			double error = layerOutputs[i] * (1 - layerOutputs[i]) * errorCausedByUnit;
			layerError.push_back(error);

			_units[i].updateWeights(error, input);
		}

		if(_loggingOn)
		{
			logLayerError(layerError);
			logUnitWeights();
		}
	}
	else
	{
		if(_loggingOn)
		{
			std::cout << "Prediction: " << vectorToString(layerOutputs) << std::endl;
		}

		for(size_t i  = 0; i < _units.size(); i++)
		{
			double error = layerOutputs[i] * (1 - layerOutputs[i]) * (input[i] - target[i]);
			layerError.push_back(error);
			_units[i].updateWeights(error, input);
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

size_t BackPropLayer::getLayerId() const
{
	return _layerId;
}

const BackPropUnit& BackPropLayer::operator[](size_t i) const
{
	assert(i < _units.size());
	return _units[i];
}

const BackPropUnit& BackPropLayer::getUnit(size_t unitIndex) const
{
	assert(unitIndex < _units.size());
	return _units[unitIndex];
}

BackPropUnit& BackPropLayer::getUnit(size_t unitIndex)
{
	assert(unitIndex < _units.size());
	return _units[unitIndex];
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
	ss << ", input layer = " << (_prevLayer == NULL ? "true" : "false");
	ss << ", output layer = " << (_nextLayer == NULL ? "true" : "false");
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