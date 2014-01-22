#ifndef _PERCEPTRON_NODE_H_
#define _PERCEPTRON_NODE_H_

#include <vector>
#include <cassert>
#include <iostream>
#include <string>
#include "matrix.h"

/**
 * A node's output
 */
struct NodeOutput
{
    /**The 1 if it exceeds the threshold and 0 if it doesn't*/
	double output;
    /**The net-output value*/
    double netOutput;
};

/**
 * Represents a single perceptron node. This is a base class.
 */
class PerceptronNode
{
	public:
		/**
		 * Default constructor
         */
        PerceptronNode();

        /**
         * Constructor
         * @param nWeight - The number of weights
         */
        PerceptronNode(size_t nFeatures, double learningRate, size_t targetLabelIndex, double targetClass);

        /**
         * Deconstructor
         */
        virtual ~PerceptronNode();

        /**
         * Updates the weights given a training input
         */
        virtual void train(Matrix& features, Matrix& labels) = 0;

        /**
         * Output given the input
         */
        NodeOutput getOutput(const std::vector<double>& input) const;

		/**
		 * Returns the target class
		 */
		double getTargetClass() const;

		/**
		 * To-string method 
         * @return 
         */
		virtual std::string toString() const = 0;

	protected:
		/**
		 * Returns target label index
         */
		size_t getTargetLabelIndex() const;

		/**
		 * Adjusts the weights by the given delta
		 */
		void adjustWeight(size_t weightIndex, double weightDelta);

		/**
		 * Gets the number of weights
		 */
		size_t getNumWeights() const;

		/**
		 * Gets the learning rate
		 */
		double getLearningRate() const;

		/**
		 * Gets the given weight
		 */
		double getWeight(size_t weightIndex) const;

	private:
        /**The weights vector for the perceptron node*/
        std::vector<double> _weights;
		/**The learning rate*/
		double _learningRate;
		/**The label index of interest*/
		size_t _targetLabelIndex;
		/**The classification of interest*/
		double _targetClass;
};

#endif