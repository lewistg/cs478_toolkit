#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#include <cassert>
#include <vector>
#include "learner.h"
#include "BackPropLayer.h"

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

    /**
     * Creates the layers 
     */
    void createLayers(const std::vector<size_t>& layerConfig);
};

/**
 * Sets up the test from the website
 */
void setupTest(BackProp& backProp);

#endif