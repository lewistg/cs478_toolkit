#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#include <cassert>
#include <vector>
#include <auto_ptr.h>
#include "learner.h"
#include "BackPropLayer.h"
#include "rand.h"
#include "BackPropLogger.h"

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

	/**
	 * Override
	 */
	virtual double measureTestSetAcc(Matrix& features, Matrix& labels, Matrix* pOutStats = NULL);

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
	/**Logger for experimental data*/
	BackPropLogger _logger;

    /**
     * Creates the layers 
     */
	void createLayers(const Matrix& features, Matrix& labels);

    /**
     * Calculates the accuracy of the given validation set
     * with the current model.
     * @param backProp
     */
    double measureAccuracy(Matrix& validationSet, Matrix& validationSetLabels, bool showNetwork = false);

	/**
	 * Measures the mean squared error of validation set 
     * @param src
     * @param srcLen
     * @param dest
     * @param destLen
     */
    double measureMse(Matrix& validationSet, Matrix& validationSetLabels);

	/**
	 * Copies the network layers
     * @param backProp
     */
	void copyLayers(const BackPropLayer src[], size_t srcLen, BackPropLayer** dest, size_t& destLen);

	/**
	 * Connects up the layers in a network
     * @param backProp
     */
	void connectLayers(BackPropLayer layers[], size_t nLayers);

	/**
	 * Gets the target network output based on the label
     * @param backProp
     */
	std::vector<double> targetNetworkOutput(const std::vector<double>& label, size_t valueCount);
};

/**
 * Sets up the test from the website
 */
void setupTest(BackProp& backProp);

#endif