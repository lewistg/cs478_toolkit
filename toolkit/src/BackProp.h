#ifndef _BACKPROP_H_
#define _BACKPROP_H_

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
	BackProp(Rand& rand, const std::vector<size_t>& layerConfig);

	/**
	 * Override 
     */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * Override
	 */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

private:
	/**Layers of the MLP. The last layer is the output layer.*/
	std::vector<BackPropLayer> _layers; 
};

#endif