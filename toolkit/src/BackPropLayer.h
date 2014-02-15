#ifndef _BACKPROPLAYER_H_
#define _BACKPROPLAYER_H_

#include <vector>
#include <cassert>
#include <string>
#include <sstream>
#include "BackPropUnit.h"

class BackPropUnit;

class BackPropLayer
{
public:
	/**
	 * Default constructor
     */
	BackPropLayer(bool loggingOn = false);

	/**
	 * Constructor for creating hidden and output layers
	 * @param nextLayer
     */
	BackPropLayer(Rand* rand, size_t nUnits, size_t layerId, bool loggingOn);

	/**
	 * Copies everything except the pointers
     * @param prevLayer
     */
	void copyLayerUnits(const BackPropLayer& layerToCopy);

	/**
	 * Sets the prey layer
     * @param nextLayer
     */
	void setPrevLayer(BackPropLayer* prevLayer);

	/**
	 * Match inputs with outputs of previous layer
     */
	void matchInputsToPrevLayer();

	/**
	 * Sets the random weights
     * @return 
     */
	void setRandomWeights();

	/**
	 * Sets the next layer
     */
	void setNextLayer(BackPropLayer* nextLayer);

	/**
	 * Gets the next layer
     */
	BackPropLayer* getNextLayer() const;
    
    /**
     * Gets the previous layer
     */
	BackPropLayer* getPrevLayer() const;

	/**
	 * Destructor
	 */
	~BackPropLayer();

	/**
	 * Trains on an example
	 * @return The error for each unit
     */
	std::vector<double> trainOnExample(const std::vector<double>& input, const std::vector<double>& target, long long iteration);

	/**
	 * Predicts 
     */
	void predict(const std::vector<double>& input, std::vector<double>& labels) const;

	/**
	 * Sets the layer's number
     */
	void setLayerId(size_t layerId);

	/**
	 * Gets the layer's id
     * @param i
     * @return 
     */
	size_t getLayerId() const;

	/**
	 * Getter for unit
	 */
	const BackPropUnit& operator[](size_t i) const;

	/**
	 * Getter for the unit
     * @param unitIndex
     * @return 
     */
	const BackPropUnit& getUnit(size_t unitIndex) const;

	/**
	 * Non-const getter for a unit
     * @return 
     */
	BackPropUnit& getUnit(size_t unitIndex);

    /**
     * Gets the number of units
     */
    size_t getNumUnits() const;

    /**
     * Sets the number of unts in the layer
     * @param nInputs
     */
    void setNumUnits(size_t nUnits);

	/**
	 * Sets the number of inputs 
     * @return 
     */
	void setNumInputs(size_t nInputs);

	/**
	 * Creates string version of this class
	 */
	std::string toString();

private:
	/**The units in the layer*/
	std::vector<BackPropUnit> _units;
	/**Pointer to the prev layer*/
	BackPropLayer* _prevLayer;
	/**Pointer to the next layer*/
	BackPropLayer* _nextLayer;
	/**The layer's id*/
	size_t _layerId;
	/**Logging flag*/
	bool _loggingOn;
	/**Random number generator*/
	Rand* _rand;

	/**
	 * Logging function for layer error during training
	 */
	void logLayerError(const std::vector<double>& layerError);

	/**
	 * Logging for unit weights
	 */
	void logUnitWeights();
};

#endif