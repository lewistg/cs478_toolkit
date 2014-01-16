#ifndef _PERCEPTRON_RULE_PERCEPTRON_NODE_H_
#define _PERCEPTRON_RULE_PERCEPTRON_NODE_H_

#include <string>
#include "perceptron_node.h"

class PerceptronRulePerceptronNode: public PerceptronNode 
{
	public:
		/**
		 * Constructor
		 */
		PerceptronRulePerceptronNode(size_t nFeatures, double learningRate, size_t targetLabelIndex, double targetClass);
		
		/**
		 * Deconstructor
		 */
		~PerceptronRulePerceptronNode();

		/**
		 * Override
		 */
		virtual void train(Matrix& features, Matrix& labels);

		/**
		 * To-string method
		 */
		std::string toString() const;

	private:
};

#endif