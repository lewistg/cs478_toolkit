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
	int nBiasWeights = 1;
	// set the weights to random values in the range [-0.05, 0.05]
	for(size_t i = 0; i < nWeights + nBiasWeights; i++)
		_weights[i] = (r.normal() * 0.1) - 0.05;
}

BackPropUnit::~BackPropUnit()
{

}

double BackPropUnit::getOutput(const std::vector<double>& features)
{
	for(size_t i = 0; i < features.size(); i++)	

	return 0.0;
}

double BackPropUnit::getWeight(size_t i )
{
	return _weights[i];
}

void BackPropUnit::updateWeights(double error)
{

}
