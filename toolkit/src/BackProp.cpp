#include <vector>
#include <cmath>
#include "BackProp.h"
#include "data_utils.h"

BackProp::BackProp(Rand& rand, bool loggingOn):_layers(NULL), _nLayers(0), _loggingOn(loggingOn), _rand(rand)
{
}

BackProp::~BackProp()
{
	if(_layers != NULL)
		delete [] _layers;
	_layers = NULL;
}

void BackProp::train(Matrix& features, Matrix& labels)
{
	createLayers(features, labels);

	if(_loggingOn)
	{
		std::cout << "Network created..." << std::endl;
		BackPropLayer* itr = &_layers[0];
		while(itr != NULL)
		{
			std::cout << itr->toString() << std::endl;
			itr = itr->getNextLayer();
		}
	}

	features.shuffleRows(_rand, &labels);

	double percentValidation = 0.25;
	size_t validationSetSize = static_cast<size_t>(std::max(percentValidation * features.rows(), 1.0));
	size_t testSetSize = features.rows() - validationSetSize;
	assert(validationSetSize > 0 && validationSetSize < features.rows());

	Matrix testSet;
	testSet.copyPart(features, 0, 0, testSetSize, features.cols());

	Matrix testSetLabels;
	testSetLabels.copyPart(labels, 0, 0, testSetSize, labels.cols());

	Matrix validationSet;
	validationSet.copyPart(features, testSetSize, 0, validationSetSize, features.cols());

	Matrix validationSetLabels;
	validationSetLabels.copyPart(labels, testSetSize, 0, validationSetSize, labels.cols());

	double bestValidationAccuracy = 0.0;
	BackPropLayer* bestSolutionSoFar = NULL;
	size_t bssfLen = 0;
	size_t epochsWithSameBssf = 0;
	

	long long iteration = 0;
	while(true)
	{
		testSet.shuffleRows(_rand, &testSetLabels);
		for(size_t i = 0; i < features.rows(); i++)
		{
			std::vector<double> targetOutput = targetNetworkOutput(labels.row(i), labels.valueCount(0));
			/*std::vector<double> targetOutput(labels.valueCount(0), 0);
			size_t positiveLabelIndex = labels.row(i)[0];
			targetOutput[positiveLabelIndex] = 1.0;*/

			if(_loggingOn)
			{
				std::cout << "Training example input vector: " << vectorToString(features.row(i)) << std::endl;
				std::cout << "Target label: " << vectorToString(labels.row(i)) << std::endl;
				std::cout << "Target output: " << vectorToString(targetOutput) << std::endl;
			}

			_layers[0].trainOnExample(features.row(i), targetOutput, iteration);
		}
		iteration++;

		double tsAcc = measureAccuracy(testSet, testSetLabels);
		double tsMse = measureMse(testSet, testSetLabels);
		double vsAcc = measureAccuracy(validationSet, validationSetLabels);
		double vsMse = measureMse(validationSet, validationSetLabels);
		EpochStats epochStats(tsAcc, tsMse, vsAcc, vsMse);
		_logger.logStats(epochStats);

		//std::cout << "Validation set accuracy: " << validationAccuracy << std::endl;
		//std::cout << "Validation set mean squared error: " << mse << std::endl;
		if(vsAcc > bestValidationAccuracy)
		{
			epochsWithSameBssf = 0;
			copyLayers(_layers, _nLayers, &bestSolutionSoFar, bssfLen);
			bestValidationAccuracy = vsAcc; 
		}
		else
		{
			epochsWithSameBssf += 1;
			if(epochsWithSameBssf > 100)
			{
				copyLayers(bestSolutionSoFar, bssfLen, &_layers, _nLayers);
				break;
			}
		}

		/*if(_loggingOn)
		{
			std::cout << "Network after epoch: " << std::endl;
			BackPropLayer* itr = &_layers[0];
			while(itr != NULL)
			{
				std::cout << itr->toString() << std::endl;
				itr = itr->getNextLayer();
			}
		}*/
	}
}

void BackProp::copyLayers(const BackPropLayer src[], size_t srcLen, BackPropLayer** dest, size_t& destLen)
{
	assert(src != NULL);
	assert(srcLen > 0);

	if(*dest != NULL)
		delete [] *dest;
	*dest = NULL;

	assert(srcLen > 0);
	*dest = new BackPropLayer[srcLen];
	destLen = srcLen;
	for(size_t i = 0; i < srcLen; i++)
	{
		(*dest)[i].setNumUnits(src[i].getNumUnits());
		(*dest)[i].setLayerId(i);
	}

	connectLayers(*dest, destLen);

	for(size_t i = 0; i < srcLen; i++)
		(*dest)[i].copyLayerUnits(src[i]);
}

void BackProp::connectLayers(BackPropLayer layers[], size_t nLayers)
{
	for(size_t i = 1; i < nLayers; i++)	
	{
		layers[i].setPrevLayer(&_layers[i - 1]);
		layers[i].matchInputsToPrevLayer();
		layers[i - 1].setNextLayer(&_layers[i]);
	}
}

std::vector<double> BackProp::targetNetworkOutput(const std::vector<double>& label, size_t valueCount)
{
	assert(label.size() == 1);

	std::vector<double> targetOutput(valueCount, 0);
	size_t positiveLabelIndex = label[0];
	targetOutput[positiveLabelIndex] = 1.0;

	return targetOutput;
}

double BackProp::measureAccuracy(Matrix& validationSet, Matrix& validationSetLabels, bool showNetwork)
{
	if(showNetwork)
	{
		std::cout << "Network... " << std::endl;
		BackPropLayer* itr = &_layers[0];
		while(itr != NULL)
		{
			std::cout << itr->toString() << std::endl;
			itr = itr->getNextLayer();
		}
	}

	double nRight = 0;
	double total = validationSet.rows();
	std::vector<double> prediction(1);
	for(size_t i = 0; i < validationSet.rows(); i++)
	{
		double actualLabel = validationSetLabels.row(i)[0];
		predict(validationSet.row(i), prediction);

		if(actualLabel == prediction[0])
			nRight += 1;
	}

	double percentRight = nRight / total;
	return percentRight;
}

double BackProp::measureMse(Matrix& validationSet, Matrix& validationSetLabels)
{
	assert(validationSetLabels.cols() == 1);

	size_t valueCount = validationSetLabels.valueCount(0);
	double sumSquaredError = 0.0;
	double nErrors = 0.0;
	for(size_t i = 0; i < validationSet.rows(); i++)
	{
		std::vector<double> targetOutput = targetNetworkOutput(validationSetLabels.row(i), valueCount);
		std::vector<double> actualOutput;
		_layers[0].predict(validationSet.row(i), actualOutput);

		assert(targetOutput.size() == actualOutput.size());
		for(size_t j = 0; j < targetOutput.size(); j++)
		{
			double error = targetOutput[j] - actualOutput[j];
			nErrors += 1;
			sumSquaredError += error * error;
		}
	}

	double meanSquaredError = sumSquaredError / nErrors;
	return meanSquaredError;
}

void BackProp::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	assert(labels.size() == 1);
	assert(_nLayers > 0);
	assert(features.size() == _layers[0].getNumUnits());

	if(_loggingOn)
	{
		std::cout << "Input features: " << vectorToString(features) << std::endl;
	}

	std::vector<double> finalLayerOutput;
	_layers[0].predict(features, finalLayerOutput);

	if(_loggingOn)
	{
		std::cout << "Prediction: " << vectorToString(finalLayerOutput) << std::endl;
	}

	size_t maxLabel = 0;
	double maxLabelOuput = -5.0;
	for(size_t labelIndex = 0; labelIndex < finalLayerOutput.size(); labelIndex++)
	{
		if(finalLayerOutput[labelIndex] > maxLabelOuput)
		{
			maxLabel = labelIndex;
			maxLabelOuput = finalLayerOutput[labelIndex];
		}
	}

	labels[0] = maxLabel;
}

double BackProp::measureTestSetAcc(Matrix& features, Matrix& labels, Matrix* pOutStats)
{
	double acc = measureAccuracy(features, labels, pOutStats);
	EpochStats stats(0.0, 0.0, 0.0, 0.0, acc);
	_logger.logStats(stats);
	return acc;
}

void BackProp::createLayers(const Matrix& features, Matrix& labels)
{
	assert(labels.cols() == 1);
	// create the layers
    std::vector<size_t> layerConfig;
	size_t nInputs = features.cols();
    layerConfig.push_back(nInputs);

	size_t firstLayer = nInputs;
    layerConfig.push_back(firstLayer);

	//size_t hiddenLayer = firstLayer * 2;
	size_t hiddenLayer = 16; 
    layerConfig.push_back(hiddenLayer);

	size_t outputUnits = 0;
	for(size_t i = 0; i < labels.cols(); i++)
		outputUnits += (labels.valueCount(i) > 0 ? labels.valueCount(i) : 1); 
	layerConfig.push_back(outputUnits);

	if(_loggingOn)
		std::cout << "Creating network of layers..." << std::endl;

	if(_layers != NULL)
		delete _layers;

	assert(layerConfig.size() > 1);
	_nLayers = layerConfig.size() - 1;
	_layers = new BackPropLayer[_nLayers];

	for(size_t i = 1; i < layerConfig.size(); i++)
	{
		assert(layerConfig[i] > 0);
		size_t layerIndex = i - 1;
        _layers[layerIndex] = BackPropLayer(&_rand, layerConfig[i], layerIndex, _loggingOn);
		if(layerIndex == 0)
		{
			size_t inputLayerConfig = 0;
			_layers[layerIndex].setNumInputs(layerConfig[inputLayerConfig]);
		}
	}

	connectLayers(_layers, _nLayers);

	for(size_t i = 0; i < _nLayers; i++)
		_layers[i].setRandomWeights();

	assert(_nLayers > 0);
	assert(features.cols() == _layers[0].getNumUnits());

}

void setupTest(BackProp& backProp)
{
	assert(backProp._nLayers == 3);

	// setup the weights
	assert(backProp._layers[0].getNumUnits() == 2);
	assert(backProp._layers[1].getNumUnits() == 2);
	assert(backProp._layers[2].getNumUnits() == 2);

	std::vector<double> w;
	w.push_back(0.2);
	w.push_back(-0.1);
	w.push_back(0.1);

	backProp._layers[0].getUnit(0).setWeights(w);

	w[0] = 0.3;
	w[1] = -0.3;
	w[2] = -0.2;
	backProp._layers[0].getUnit(1).setWeights(w);

	w[0] = -0.2;
	w[1] = -0.3;
	w[2] = 0.1;
	backProp._layers[1].getUnit(0).setWeights(w);

	w[0] = -0.1;
	w[1] = 0.3;
	w[2] = 0.2;
	backProp._layers[1].getUnit(1).setWeights(w);

	w[0] = -0.1;
	w[1] = 0.3;
	w[2] = 0.2;
	backProp._layers[2].getUnit(0).setWeights(w);

	w[0] = -0.2;
	w[1] = -0.3;
	w[2] = 0.1;
	backProp._layers[2].getUnit(1).setWeights(w);
}