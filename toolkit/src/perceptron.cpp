#include <vector>
#include "perceptron.h"
#include "error.h"

Perceptron::Perceptron():SupervisedLearner()
{
}

Perceptron::~Perceptron()
{

}

void Perceptron::train(Matrix& features, Matrix& labels)
{
	if(features.rows() != labels.rows())
		ThrowError("Expected the features and labels to have the same number of rows");

	// create a list of perceptrons for each label
	for(size_t i = 0; i < labels.cols(); i++)
	{
		size_t values = labels.valueCount(i);
		if(values == 0) // if the label is continuous...
		{
			ThrowError("Perceptron does not support continuous labels");
		}
		else
		{
			// create a list of perceptrons
			std::vector<PerceptronRulePerceptronNode> classNodes;
			for(size_t j = 0; j < values; j++)
			{
				PerceptronRulePerceptronNode node(features.cols(), 0.1, i, static_cast<double>(j));
				classNodes.push_back(node);
			}
			
			_labelIndexToNodes.push_back(classNodes);
		}
	}

	// train all of the perceptron nodes on all the data
	for(size_t i = 0; i < features.rows(); i++)	
	{
		for(size_t j = 0; j < labels.cols(); j++)	
		{
			assert(j < _labelIndexToNodes.size());
			for(size_t k = 0; k < _labelIndexToNodes[j].size(); k++)
			{
				_labelIndexToNodes[j][k].train(features, labels);
			}
		}
	}
}

void Perceptron::predict(const std::vector<double>& features, std::vector<double>& labels)
{

}