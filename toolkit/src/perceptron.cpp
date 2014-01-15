#include <vector>
#include "perceptron.h"
#include "error.h"

void Perceptron::train(Matrix& features, Matrix& labels)
{
	if(features.rows() != labels.rows())
		ThrowError("Expected the features and labels to have the same number of rows");

	// create a list of perceptrons for each label
	for(size_t i = 0; i < labels.cols(); i++)
	{
		size_t values = labels.valueCount(i);
		if(values == 0) // if the label is continuous...
			ThrowError("Perceptron does not support continuous labels");
		else
			// create a list of perceptrons
			std::vector<PerceptronRulePerceptronNode> classNodes;
			//m_labelVec.push_back(labels.mostCommonValue(i));
	}

}

void Perceptron::predict(const std::vector<double>& features, std::vector<double>& labels)
{

}