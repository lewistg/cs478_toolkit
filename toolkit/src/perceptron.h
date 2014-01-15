#ifndef _PERCEPTRON_H_
#define _PERCEPTRON_H_

#include <vector>
#include "learner.h"
#include "perceptron_rule_perceptron_node.h"

class Perceptron : public SupervisedLearner
{
	public:	
		/**
		 * Constructor
         * @param r
         * @return 
         */
		Perceptron();

		/**
		 * Deconstructor
         */
		virtual ~Perceptron();

		/**
		 * Trains the model to predict the labels
		 */
		virtual void train(Matrix& features, Matrix& labels);

		/**
		 * Evaluate the features and predict the labels
         * @param features
         * @param labels
         */
		virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

	private:
        /**Map from label index to list of perceptrons*/
       	std::vector< std::vector<PerceptronRulePerceptronNode> > _labelIndexToNodes;

		/** 
		 * Creates the list of perceptrons
		 */
};

#endif