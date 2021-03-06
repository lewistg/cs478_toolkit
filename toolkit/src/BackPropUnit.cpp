#include <cmath>
#include <cassert>
#include <iostream>
#include <ctime>
#include "BackPropUnit.h"
#include "data_utils.h"

BackPropUnit::BackPropUnit(Rand* rand, double learningRate, double momentum, bool loggingOn):
	_trainState(true),
	_learningRate(learningRate),
	_loggingOn(loggingOn),
	_rand(rand),
	_momentum(momentum)
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
    {
		_weights.push_back(0.0);
        _prevWeightDelta.push_back(0.0);
    }
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

void BackPropUnit::updateWeights(double error, const std::vector<double>& inputs, long long iteration)
{
	assert(iteration >= 0);
	assert(inputs.size() == _weights.size() - 1);
	assert(inputs.size() == _prevWeightDelta.size() - 1);

    //std::cout << "Unit learning rate: " << _learningRate << std::endl;
    //std::cout << "Momentum: " << _momentum << std::endl;
	for(size_t i = 0; i < inputs.size(); i++)
	{
		double weightDelta = _learningRate * error * inputs[i];
		if(iteration > 0)
			weightDelta += (_prevWeightDelta[i] * _momentum);

		_prevWeightDelta[i] = weightDelta;
		_weights[i] += weightDelta;
	}

	double threshHoldWeightDelta = _learningRate * error;
	if(iteration > 0)
		threshHoldWeightDelta += _prevWeightDelta[_weights.size() - 1] 	* _momentum;

	_weights[_weights.size() - 1] += threshHoldWeightDelta;
	_prevWeightDelta[_weights.size() - 1]  = threshHoldWeightDelta;	
}

void BackPropUnit::setWeights(const std::vector<double>& weights)
{
	assert(weights.size() == _weights.size());
	_weights = weights;
}

void BackPropUnit::setRandomWeights()
{
	assert(_rand != NULL);
	for(size_t i = 0; i < _weights.size(); i++)
		_weights[i] = (_rand->normal() * 0.1);
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