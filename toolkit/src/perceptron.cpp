#include <vector>
#include "perceptron.h"
#include "error.h"
#include <float.h>

Perceptron::Perceptron(Rand& rand):_rand(rand)
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
	features.shuffleRows(_rand, &labels);
	for(size_t epochs = 0; epochs < 1; epochs++)
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
	static size_t call = 0;
	call +=1;

	for(size_t j = 0; j < labels.size(); j++)	
	{
		assert(j < _labelIndexToNodes.size());
		double label = 0.0;
		double maxNetValue = -DBL_MAX;
		for(size_t k = 0; k < _labelIndexToNodes[j].size(); k++)
		{
			NodeOutput output = _labelIndexToNodes[j][k].getOutput(features);
			std::cout << _labelIndexToNodes[j][k].toString() << " predicts " << output.output 
					<< " with net value "  << output.netOutput << std::endl;
			if(output.output == 1.0 && output.netOutput > maxNetValue)
			{
				label = static_cast<double>(k);
				maxNetValue = output.netOutput;
			}
		}
		labels[j] = label;
	}
	std::cout << std::endl;
}