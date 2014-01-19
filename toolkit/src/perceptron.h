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
		Perceptron(Rand& rand);

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

        /**
         * Gets the amount of epochs it took to train
         */
        long long getEpochsToTrain() const;

	private:
		/** Pseudo-random number generator */ 
		Rand& _rand; 
        /**Map from label index to list of perceptrons. Essentially this stores the model*/
       	std::vector< std::vector<PerceptronRulePerceptronNode> > _labelIndexToNodes;
        /**The epochs that it takes to train*/
        long long _epochsToTrain;

		/**
		 * Helper method for creating perceptrons
		 */
		void createPerceptronNodes(Matrix& features, Matrix& labels);

        /**
         * Prints out current training stats
         */
        void outputCurrStats(double accuracy, Matrix& stats) const;

		/**
		 * Prints teh current model
		 */
		void outputCurrModel() const;

		/**
		 * Binary prediction
		 * @return The label
		 */
		double getBinaryPrediction(const std::vector<double>& features, const PerceptronNode& binaryNode) const;

		/**
		 * Gets the prediction based on a list of nodes. This is for when the label could
		 * more than one thing.
		 */
		double getMulitNodePrediction(const std::vector<double>& features, 
				const std::vector<PerceptronRulePerceptronNode>& nodes) const;

		/**
		 * Signals training to stop once 5 epochs have passed without
		 * changing accuracy more than 1 percent.
		 */
		bool accuracyNotChanging(long long epochsTrainedSoFar, Matrix& features, Matrix& labels);

		/**
		 * Signals training to stop once 100 epochs have passed without
		 * improvement upon the best model that has been found.
		 */
		bool bestModelNotImproved(long long epochsTrainedSoFar, 
				Matrix& features, Matrix& labels, 
				std::vector< std::vector<PerceptronRulePerceptronNode> >& bestModel);
};

#endif