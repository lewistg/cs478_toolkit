#ifndef _BACKPROPLAYER_H_
#define _BACKPROPLAYER_H_

#include <vector>
#include <cassert>
#include "BackPropUnit.h"

class BackPropUnit;

class BackPropLayer
{
public:
	/**
	 * Default constructor
     */
	BackPropLayer();

	/**
	 * Constructor for creating hidden and output layers
	 * @param nextLayer
     */
	BackPropLayer(size_t nUnits);

	/**
	 * Sets the prey layer
     * @param nextLayer
     */
	void setPrevLayer(BackPropLayer* prevLayer);

	/**
	 * Sets the next layer
     */
	void setNextLayer(BackPropLayer* nextLayer);

	/**
	 * Destructor
	 */
	~BackPropLayer();

	/**
	 * Trains on an example
	 * @return The error for each unit
     */
	std::vector<double> trainOnExample(const std::vector<double>& input, const std::vector<double>& target);

	/**
	 * Getter for unit
	 */
	const BackPropUnit& operator[](size_t i);

private:
	/**The units in the layer*/
	std::vector<BackPropUnit> _units;
	/**Pointer to the prev layer*/
	BackPropLayer* _prevLayer;
	/**Pointer to the next layer*/
	BackPropLayer* _nextLayer;
};

#endif