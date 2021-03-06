#include <iostream>
#include "data_utils.h"

void partitionTrainAndVal(Matrix& features, Matrix& labels, 
		Matrix& trainingSet, Matrix& trainingSetLabels,
		Matrix& validationSet, Matrix& validationSetLabels)
{
	double percentValidation = 0.25;
	size_t validationSetSize = static_cast<size_t>(std::max(percentValidation * features.rows(), 1.0));
	size_t testSetSize = features.rows() - validationSetSize;
	assert(validationSetSize > 0 && validationSetSize < features.rows());

	trainingSet.copyPart(features, 0, 0, testSetSize, features.cols());
	trainingSetLabels.copyPart(labels, 0, 0, testSetSize, labels.cols());

	validationSet.copyPart(features, testSetSize, 0, validationSetSize, features.cols());
	validationSetLabels.copyPart(labels, testSetSize, 0, validationSetSize, labels.cols());
} 

std::string getInstanceString(const std::vector<double>& instance, Matrix& features)
{
	std::stringstream ss;
	for(size_t i = 0; i < features.cols(); i++)
	{
		if(instance[i] == UNKNOWN_VALUE)
			ss << "?";
		else if(features.valueCount(i) != 0)
			ss << features.attrValue(i, instance[i]);
		else
			ss << instance[i];

		if(i < features.cols())
			ss << ", ";
	}

	return ss.str();
}