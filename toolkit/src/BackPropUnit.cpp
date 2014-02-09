#include <cmath>
#include <cassert>
#include <iostream>
#include <ctime>
#include "BackPropUnit.h"
#include "data_utils.h"

BackPropUnit::BackPropUnit(Rand& rand, bool loggingOn):
	_trainState(true),
	_learningRate(0.1),
	_loggingOn(loggingOn),
	_rand(&rand)
{

}

BackPropUnit::~BackPropUnit()
{

}

void BackPropUnit::setNumInputs(size_t nInputs)
{
	if(!_weights.empty())
		_weights.clear();

	int nBiasWeights = 1;
	assert(nInputs > 0);
	for(size_t i = 0; i < nInputs + nBiasWeights; i++)
		_weights.push_back((_rand->normal() * 0.1) - 0.05);
}	

double BackPropUnit::getOutput(const std::vector<double>& features) const
{
	if(_loggingOn)
	{
		std::cout << "Unit weights: " << vectorToString(_weights) << std::endl;
	}

	double sum = 0.0;
	assert(features.size() == _weights.size() - 1);
	for(size_t i = 0; i < features.size(); i++)	
		sum += _weights[i] * features[i];

	sum += _weights[_weights.size() - 1];

	double output = 1 / (1 + exp(-sum));
	return output; 
}

double BackPropUnit::getWeight(size_t i ) const
{
	return _weights[i];
}

void BackPropUnit::updateWeights(double error, const std::vector<double>& inputs)
{
	assert(inputs.size() == _weights.size() - 1);
	for(size_t i = 0; i < inputs.size(); i++)
	{
		double weightDelta = _learningRate * error * inputs[i];
		_weights[i] += weightDelta;
	}

	double threshHoldWeightDelta = _learningRate * error;
	_weights[_weights.size() - 1] += threshHoldWeightDelta;
}

void BackPropUnit::setWeights(const std::vector<double>& weights)
{
	assert(weights.size() == _weights.size());
	_weights = weights;
}

std::string BackPropUnit::toString()
{
	std::stringstream ss;
	ss << "(Weights = [";
	for(size_t i = 0; i < _weights.size(); i++)
	{
		ss << _weights[i];
		if(i < _weights.size() - 1)
			ss << ", ";
		else
			ss << "]";
	}

	return ss.str();
}