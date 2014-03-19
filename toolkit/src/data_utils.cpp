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
