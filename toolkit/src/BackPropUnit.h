#ifndef _BACKPROPUNIT_H_
#define _BACKPROPUNIT_H_

#include <cassert>
#include <vector>
#include <string>
#include <sstream>
#include "perceptron_node.h"
#include "rand.h"

class BackPropLayer;

class BackPropUnit
{
public:
	/**
	 * Default constructor
     */
	BackPropUnit(Rand& rand, bool loggingOn = false);

	/**
	 * Deconstructor
	 */
	~BackPropUnit();

	/**
	 * Sets the number of inputs that are used 
     * @param features
     * @return 
     */
	void setNumInputs(size_t nInputs);

	/**
	 * Trains using the given example
	 * @return The net value
	 */
	double getOutput(const std::vector<double>& features) const;

	/**
	 * Gets a weight
     * @param i
     * @return 
     */
	double getWeight(size_t i ) const;

	/**
	 * Updates weights given the error term associated with the given
	 * features
     * @pre These are the same weights passed into the train on example
	 */
	void updateWeights(double error, const std::vector<double>& inputs);

	/**
	 * Sets the weights
	 * @pre The number of weights matches the number of weighs already created
     * @return 
     */
	void setWeights(const std::vector<double>& weights);

	/**
	 * Sets the weights randomly
     * @return 
     */
	void setRandomWeights();

	/**
	 * Gets string containing unit info
	 */
	std::string toString();

private:
	/**Keeps track of whether or not we are training or updating weights*/
	bool _trainState;
	/**Weight vector*/
	std::vector<double> _weights;
	/**Learning rate*/
	double _learningRate;
	/**Logging on or off*/
	bool _loggingOn;
	/**Random number generator*/
	Rand* _rand;
};

#endif