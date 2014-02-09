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

	/*double percentValidation = 0.25;
	size_t validationSetSize = static_cast<size_t>(std::max(percentValidation * features.rows(), 1.0));
	size_t testSetSize = features.rows() - validationSetSize;
	assert(validationSetSize > 0 && validationSetSize < features.rows());*/

	/*Matrix testSet;
	Matrix testSetLabels;

	//copyPart(Matrix& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount)
	features.copyPart(testSet, 0, 0, testSetSize, features.cols());
	labels.copyPart(testSetLabels, 0, 0, testSetSize, labels.cols());

	Matrix validationSet;
	Matrix validationSetLabels;
	features.copyPart(validationSet, testSetSize, 0, validationSetSize, features.cols());
	labels.copyPart(validationSetLabels, testSetSize, 0, validationSetSize, labels.cols());*/

	//double bestValidationAccuracy = 0.0;
	//BackProp* bestLayers = NULL;
	

	for(size_t epochs = 0; epochs < 1000; epochs++)
	{
		features.shuffleRows(_rand, &labels);
		for(size_t i = 0; i < features.rows(); i++)
		{
			std::vector<double> targetOutput(labels.valueCount(0), 0);
			size_t positiveLabelIndex = labels.row(i)[0];
			targetOutput[positiveLabelIndex] = 1.0;

			if(_loggingOn)
			{
				std::cout << "Training example input vector: " << vectorToString(features.row(i)) << std::endl;
				std::cout << "Target label: " << vectorToString(labels.row(i)) << std::endl;
				std::cout << "Target output: " << vectorToString(targetOutput) << std::endl;
			}

			_layers[0].trainOnExample(features.row(i), targetOutput);
		}

		/*double validationAccuracy = measureAccuracy(validationSet, validationSetLabels);
		if(validationAccuracy < bestValidationAccuracy)
		{
			
		}*/

		if(_loggingOn)
		{
			std::cout << "Network after epoch: " << std::endl;
			BackPropLayer* itr = &_layers[0];
			while(itr != NULL)
			{
				std::cout << itr->toString() << std::endl;
				itr = itr->getNextLayer();
			}
		}
	}
}

void BackProp::copyLayers(BackPropLayer*& layerCopy)
{
	if(layerCopy != NULL)
		delete [] layerCopy;
	layerCopy = NULL;

	assert(_nLayers > 0);
	layerCopy = new BackPropLayer[_nLayers];
	for(size_t i = 0; i < _nLayers; i++)
		layerCopy[i].copyLayerUnits(_layers[i]);
}

void BackProp::connectLayers(BackPropLayer layers[], size_t nLayers)
{
	for(size_t i = 1; i < nLayers; i++)	
	{
		layers[i].setPrevLayer(&_layers[i - 1]);
		layers[i - 1].setNextLayer(&_layers[i]);
	}
}

double BackProp::measureAccuracy(Matrix& validationSet, Matrix& validationSetLabels)
{
	double nRight = 0;
	double total = validationSet.rows();
	std::vector<double> prediction;
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

void BackProp::createLayers(const Matrix& features, Matrix& labels)
{
	assert(labels.cols() == 1);
	// create the layers
    std::vector<size_t> layerConfig;
	size_t nInputs = features.cols();
    layerConfig.push_back(nInputs);

	size_t firstLayer = nInputs;
    layerConfig.push_back(firstLayer);

	size_t hiddenLayer = firstLayer * 4;
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
        _layers[layerIndex] = BackPropLayer(_rand, layerConfig[i], layerIndex, _loggingOn);
		if(layerIndex == 0)
		{
			size_t inputLayerConfig = 0;
			_layers[layerIndex].setNumInputs(layerConfig[inputLayerConfig]);
		}
	}

	connectLayers(_layers, _nLayers);

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