#include "perceptron_rule_perceptron_node.h"

PerceptronRulePerceptronNode::PerceptronRulePerceptronNode(size_t nFeatures, 
			double learningRate, 
			size_t targetLabelIndex, 
			double targetClass):
	PerceptronNode(nFeatures, learningRate, targetLabelIndex, targetClass)
{

}

PerceptronRulePerceptronNode::~PerceptronRulePerceptronNode()
{

}

void PerceptronRulePerceptronNode::train(Matrix& features, Matrix& labels)
{
	size_t targetLabel = getTargetLabelIndex();	
	assert(targetLabel < labels.cols());

	double targetClass = getTargetClass();
	for(size_t i = 0; i < features.rows(); i++)
	{
		double targetVal = 0;
		if(labels.row(i)[targetLabel] == targetClass)
			targetVal = 1;

		NodeOutput output = getOutput(features.row(i));
		for(size_t j = 0; j < getNumWeights(); j++)
		{
			double x_j = 0.0; 
			if(j == getNumWeights() - 1)
				x_j = 1.0; // the bias input is always 1
			else
				x_j = features.row(i)[j];

			double weightDelta = getLearningRate() * (targetVal - output.output) * x_j;
			adjustWeight(j, weightDelta);
		}
	}
}