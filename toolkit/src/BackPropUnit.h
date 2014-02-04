#ifndef _BACKPROPUNIT_H_
#define _BACKPROPUNIT_H_

#include <cassert>
#include <vector>
#include "perceptron_node.h"

class BackPropLayer;

class BackPropUnit
{
public:
	/**
	 * Default constructor
     */
	BackPropUnit();

	/**
	 * Constructor
	 */
	BackPropUnit(Rand r, size_t nWeights);

	/**
	 * Deconstructor
	 */
	~BackPropUnit();

	/**
	 * Trains using the given example
	 * @return The net value
	 */
	double getOutput(const std::vector<double>& features);

	/**
	 * Gets a weight
     * @param i
     * @return 
     */
	double getWeight(size_t i );

	/**
	 * Updates weights given the error term associated with the given
	 * features
     * @pre These are the same weights passed into the train on example
	 */
	void updateWeights(double error);

private:
	/**Keeps track of whether or not we are training or updating weights*/
	bool _trainState;
	/**Weight vector*/
	std::vector<double> _weights;
	/**Learning rate*/
	double _learningRate;
};

#endif