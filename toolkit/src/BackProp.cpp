#include <vector>
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

	createLayers(layerConfig);

	//setupTest(*this);

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

	assert(_nLayers > 0);
	assert(features.cols() == _layers[0].getNumUnits());

	for(size_t epochs = 0; epochs < 1000; epochs++)
	{
		features.shuffleRows(_rand, &labels);
		for(size_t i = 0; i < features.rows(); i++)
		{
			std::vector<double> targetOutput(outputUnits, 0);
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

void BackProp::createLayers(const std::vector<size_t>& layerConfig)
{
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
		else
		{

			_layers[layerIndex].setPrevLayer(&_layers[layerIndex - 1]);
			/*if(_loggingOn)
			{
				std::cout << "Layer " << _layers[layerIndex].getLayerId() << 
						"'s prev is now " << _layers[layerIndex].getPrevLayer()->toString() << std::endl;
			}*/

			_layers[layerIndex - 1].setNextLayer(&_layers[layerIndex]);
			/*if(_loggingOn)
			{
				std::cout << layerIndex - 1 << std::endl;
				std::cout << "Layer " << _layers[layerIndex-1].getLayerId() << "'s next layer is now " <<
						_layers[layerIndex - 1].getNextLayer()->toString() << std::endl;	
			}*/

			/*if(_loggingOn)
			{
				std::cout << "Network so far..." << std::endl;
				BackPropLayer* itr = &_layers[0];
				while(itr != NULL)
				{
					std::cout << itr->toString() << std::endl;
					itr = itr->getNextLayer();
				}
			}*/
		}
	}
	assert(_layers[1].getNextLayer() != NULL);

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