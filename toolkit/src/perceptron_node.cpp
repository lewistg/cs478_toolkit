#include "perceptron_node.h"

PerceptronNode::PerceptronNode(size_t nFeatures, double learningRate, size_t targetLabelIndex, double targetClass):
	_weights(nFeatures + 1), // add one for the bias weight
	_learningRate(learningRate),
	_targetLabelIndex(targetLabelIndex),
	_targetClass(targetClass)
{
    assert(nFeatures > 0);
	assert(learningRate > 0);
}

PerceptronNode::~PerceptronNode()
{

}

size_t PerceptronNode::getTargetLabelIndex() const
{
	return _targetLabelIndex;
}

double PerceptronNode::getTargetClass() const
{
	return _targetClass;
}

void PerceptronNode::adjustWeight(size_t weightIndex, double weightDelta)
{
	assert(weightIndex < _weights.size());
	_weights[weightIndex] += weightDelta;
}

size_t PerceptronNode::getNumWeights() const
{
	return _weights.size();
}

double PerceptronNode::getLearningRate() const
{
	return _learningRate;
}

double PerceptronNode::getWeight(size_t weightIndex) const
{
	assert(weightIndex < _weights.size());
	return _weights[weightIndex];
}

NodeOutput PerceptronNode::getOutput(const std::vector<double>& input) const
{
	assert(input.size() == _weights.size() - 1); // recall the bias is handled implicitly

	// the bias input is handled implicitly
	double netValue = 0.0;
	size_t i = 0;
	for(i = 0; i < input.size(); i++)
		netValue += input[i] * _weights[i];

	// implicitly handle the bias input (recall that it is 1, thus
	// _weights[i] * 1.0 = _weights[i]
	assert(i == input.size());
	netValue += _weights[i];

	NodeOutput output;
	output.netOutput = netValue;
	output.output = (netValue > 0 ? 1 : 0);

	return output;
}