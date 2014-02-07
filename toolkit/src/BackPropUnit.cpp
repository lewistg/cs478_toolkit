#include <cmath>
#include <cassert>
#include "BackPropUnit.h"
#include "rand.h"

BackPropUnit::BackPropUnit()
{

}

BackPropUnit::BackPropUnit(Rand r, size_t nWeights):
	_trainState(true),
	_weights(nWeights + 1), // add one for the bias weight
	_learningRate(0.1)
{
	assert(nWeights > 0);
	int nBiasWeights = 1;
	// set the weights to random values in the range [-0.05, 0.05]
	for(size_t i = 0; i < nWeights + nBiasWeights; i++)
		_weights[i] = (r.normal() * 0.1) - 0.05;
}

BackPropUnit::~BackPropUnit()
{

}

double BackPropUnit::getOutput(const std::vector<double>& features) const
{
	double sum = 0.0;
	assert(features.size() == _weights.size() - 1);
	for(size_t i = 0; i < features.size() - 1; i++)	
		sum += _weights[i] * features[i];

	sum += _weights[_weights.size() - 1];

	double output = 1 / (1 + exp(-sum));
	return output; 
}

double BackPropUnit::getWeight(size_t i )
{
	return _weights[i];
}

void BackPropUnit::updateWeights(double error)
{

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