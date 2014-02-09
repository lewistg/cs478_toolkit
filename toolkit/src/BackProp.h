#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#include <cassert>
#include <vector>
#include <auto_ptr.h>
#include "learner.h"
#include "BackPropLayer.h"
#include "rand.h"

/**
 * Implements the Back Propogation algorithm
 */
class BackProp: public SupervisedLearner
{
public:
	/**
	 * Constructor
	 */
	BackProp(Rand& rand, bool loggingOn = false);

	/**
	 * Destructor
     */
	~BackProp();

	/**
	 * Override 
     */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * Override
	 */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

	friend void setupTest(BackProp& backProp);

private:
	/**Layers of the MLP. The last layer is the output layer.*/
	BackPropLayer* _layers; 
	/**The number of layers*/
	size_t _nLayers;
	/**Turns on logging or not*/
	bool _loggingOn;
	/**Random num gnerator*/
	Rand _rand;

    /**
     * Creates the layers 
     */
	void createLayers(const Matrix& features, Matrix& labels);

    /**
     * Calculates the accuracy of the given validation set
     * with the current model.
     * @param backProp
     */
    double measureAccuracy(Matrix& validationSet, Matrix& validationSetLabels);

	/**
	 * Copies the network layers
     * @param backProp
     */
	void copyLayers(BackPropLayer*& layerCopy);

	/**
	 * Connects up the layers in a network
     * @param backProp
     */
	void connectLayers(BackPropLayer layers[], size_t nLayers);
};

/**
 * Sets up the test from the website
 */
void setupTest(BackProp& backProp);

#endif