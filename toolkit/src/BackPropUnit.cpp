#include <cmath>
#include <cassert>
#include "rand.h"
#include "BackPropUnit.h"
#include "rand.h"

BackPropUnit::BackPropUnit():
	_trainState(true),
	_learningRate(0.1)
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
	Rand r(56);
	for(size_t i = 0; i < nInputs + nBiasWeights; i++)
		_weights.push_back((r.normal() * 0.1) - 0.05);
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