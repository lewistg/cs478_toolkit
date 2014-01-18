#include <vector>
#include "perceptron.h"
#include "error.h"
#include <float.h>
#include <cmath>
#include <sstream>

namespace 
{
	bool DEBUG = true;
    bool TRAINING_STATS = true;
}

Perceptron::Perceptron(Rand& rand):
	_rand(rand),
	_epochsToTrain(0)
{
}

Perceptron::~Perceptron()
{

}

void Perceptron::createPerceptronNodes(Matrix& features, Matrix& labels)
{
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
			bool isBinary = (values == 2);
			for(size_t j = 0; j < values; j++)
			{
				PerceptronRulePerceptronNode node(features.cols(), 0.1, i, static_cast<double>(j));
				classNodes.push_back(node);

				// only one node is necessary for a binary label
				if(isBinary)
					break;
			}
			
			_labelIndexToNodes.push_back(classNodes);
		}
	}
}

void Perceptron::train(Matrix& features, Matrix& labels)
{
	if(features.rows() != labels.rows())
		ThrowError("Expected the features and labels to have the same number of rows");

	createPerceptronNodes(features, labels);
	_epochsToTrain = 0;

	// train all of the perceptron nodes on all the data
	double prevAccuracy = 0.0;
	double currAccuracy = 0.0;
	size_t epochsWithoutChange = 0;
	while(true)
	{
		features.shuffleRows(_rand, &labels);
		for(size_t j = 0; j < labels.cols(); j++)	
		{
			assert(j < _labelIndexToNodes.size());
			for(size_t k = 0; k < _labelIndexToNodes[j].size(); k++)
			{
				_labelIndexToNodes[j][k].train(features, labels);
			}
		}

		_epochsToTrain += 1;

		// decide if we should stop training or not
		Matrix stats;
		currAccuracy = measureAccuracy(features, labels, &stats);
		if(TRAINING_STATS)
			outputCurrStats(currAccuracy, stats);

		if(fabs(currAccuracy - prevAccuracy) < 0.01)
		{
			if(epochsWithoutChange < 5)
			{

				epochsWithoutChange += 1;
			}
			else
			{
				if(TRAINING_STATS)
					outputCurrModel();
				return;
			}
		}
		else
		{
				epochsWithoutChange = 0;
		}
		prevAccuracy = currAccuracy;
        if(DEBUG)
			std::cout << currAccuracy << std::endl;
	}
}

void Perceptron::outputCurrStats(double accuracy, Matrix& stats) const
{
	std::stringstream outputLabel;
	std::stringstream dataOutput;
	outputLabel << "Current epoch\t";
	dataOutput << _epochsToTrain << "\t";

	outputLabel << "Set accuracy\t";
	dataOutput << accuracy << "\t";

	outputLabel << "Set error rate\t";
	dataOutput << 1.0 - accuracy << "\t";

	for(size_t i = 0; i < stats.cols(); i++)
	{
		outputLabel << "Label #: " << i << "\t";
		dataOutput << stats[0][i] << "/" << stats[1][i] << "\t";
	}

	std::cout << outputLabel.str() << std::endl;	
	std::cout << dataOutput.str() << std::endl;
}

void Perceptron::outputCurrModel() const
{
	std::cout << "Final model" << std::endl;
	for(size_t i = 0; i < _labelIndexToNodes.size(); i++)
	{
		for(size_t j = 0; j < _labelIndexToNodes[i].size(); j++)
			std::cout << _labelIndexToNodes[i][j].toString() << std::endl;
	}
	
}

void Perceptron::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	static size_t call = 0;
	call +=1;

	for(size_t j = 0; j < labels.size(); j++)	
	{
		assert(j < _labelIndexToNodes.size());
		bool binaryLabel = (_labelIndexToNodes.size() == 1);
		if(binaryLabel)
			labels[j] = getBinaryPrediction(features, _labelIndexToNodes[j][0]);
		else
			labels[j] = getMulitNodePrediction(features, _labelIndexToNodes[j]);
	}
}

double Perceptron::getBinaryPrediction(const std::vector<double>& features, const PerceptronNode& binaryNode) const
{
	NodeOutput output = binaryNode.getOutput(features);
	assert(binaryNode.getTargetClass() == 0.0);
	if(output.output == 0.0)
		return 1.0;
	else
		return 0.0;
}

double Perceptron::getMulitNodePrediction(const std::vector<double>& features, 
		const std::vector<PerceptronRulePerceptronNode>& nodes) const
{
	double labelPos = 0.0;
	double maxNetValueForPos = -DBL_MAX;
	double labelNeg = 0.0;
	double maxNetValueForNeg = -DBL_MAX;
	bool allNeg = true;
	for(size_t k = 0; k < nodes.size(); k++)
	{
		NodeOutput output = nodes[k].getOutput(features);
		if(DEBUG)
		{
			std::cout << "\t" << nodes[k].toString() << " predicts " << output.output 
					<< " with net value "  << output.netOutput << std::endl;
		}

		if(output.output == 1.0 && output.netOutput > maxNetValueForPos)
		{
			allNeg = false;
			labelPos = static_cast<double>(nodes[k].getTargetClass());
			maxNetValueForPos = output.netOutput;
		}
		else if(output.output == 0 && output.netOutput > maxNetValueForNeg)
		{
			labelNeg = 	static_cast<double>(nodes[k].getTargetClass());
			maxNetValueForNeg = output.netOutput;
		}
	}

	if(allNeg)
		return labelNeg;
	else
		return labelPos;
}

long long Perceptron::getEpochsToTrain() const
{
	return _epochsToTrain;
}