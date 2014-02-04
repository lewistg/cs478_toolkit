#include <sstream>
#include "perceptron_rule_perceptron_node.h"
#include "data_utils.h"

namespace
{
	bool DEBUG = false; 
}

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
	size_t targetLabelIndex = getTargetLabelIndex();	
	assert(targetLabelIndex < labels.cols());

	double targetClass = getTargetClass();
	for(size_t i = 0; i < features.rows(); i++)
	{
		double targetVal = 0;
		if(labels.row(i)[targetLabelIndex] == targetClass)
			targetVal = 1;

		if(DEBUG/* && targetClass == 2*/)
		{
			std::cout << "Before example: " << toString() << std::endl;
			std::cout << "Pattern: " << vectorToString<double>(features.row(i)) << std::endl;
			std::cout << "Label: " << labels.row(i)[targetLabelIndex] << std::endl;
		}

		NodeOutput output = getOutput(features.row(i));
		if(DEBUG/* && targetClass ==  2*/)
		{
			std::cout << "Node predicted: " << output.output << std::endl;
			std::cout << "Net value of: " << output.netOutput << std::endl;
		}
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

		if(DEBUG/* && targetClass == 2*/)
			std::cout << "After example: " << toString() << std::endl << std::endl;
	}
}

std::string PerceptronRulePerceptronNode::toString() const
{
	std::stringstream ss;
	ss << "(Target label index = " << getTargetLabelIndex() << ", ";
	ss << "Target class = " << getTargetClass() << ", ";
	for(size_t i = 0; i < getNumWeights(); i++)
	{
		ss << "w_"	<< i << " = " << getWeight(i); 
		if(i < getNumWeights() - 1)
			ss << ", ";
	}
	ss << ")";
	
	return ss.str();
}