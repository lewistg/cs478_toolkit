#ifndef _KNEARESTNEIGHBOR_H_
#define _KNEARESTNEIGHBOR_H_

#include <queue>
#include "learner.h"

//struct PairCmpr;
struct PairCmpr
{
	bool operator() (const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) 
	{
		return lhs.second < rhs.second;
	}
};

class KNearestNeighbor: public SupervisedLearner
{
public:
	/**
	 * Constructor
     * @param features
     * @param labels
     */
	KNearestNeighbor();

	/**
	 * @override
     */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * @override
     */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

private:
	/**The number of neighbors to include*/
	size_t _k;
	/**Nearest neighbors*/
	Matrix _examples;
	/**The neighbor labels*/
	Matrix _exampleLabels;
	/**The original min and max points*/
	std::vector<double> _colMaxes;
	std::vector<double> _colMins;

	double _minLabel;
	double _maxLabel;

	/**
	 * Does prediction ignoring certain training examples
     */
	void predictWithIgnore(const std::vector<double>& features, std::vector<double>& labels, 
		std::vector<bool>* ignoredExamples = NULL);

	/**
	 * Utility function for calculating the distance between a training example
	 * and the given features.
	 */
	double dist(const std::vector<double>& features, const std::vector<double>& example);

    /**
     * Calculates the heom distance between two points
     * @param features
     * @return 
     */
	double heom(const std::vector<double>& features, const std::vector<double>& example);

	/**
	 * Calculates the Manhattan distance between two points
     * @param features
     */
	double manhattanDist(const std::vector<double>& features, const std::vector<double>& example);

    /**
     * Normalizes the training example
     */
    void normalizeFeatures(std::vector<double>& features);

	/**
	 * Does prediction using nominal data
	 */
	void predictNominal(const std::vector<double>& normFeatures, std::vector<double>& labels, 
		std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr>&
		nearestKInstances);

	/**
	 * Does regression
	 */
	void regressionPrediction(const std::vector<double>& normFeatures, std::vector<double>& labels, 
		std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr>&
		nearestKInstances);

	/**
	 * Test set accuracy
     * @return 
     */
	//double measureTestSetAcc(Matrix& features, Matrix& labels, Matrix* pOutStats = NULL);

    /**
     * Does leave one out reduction
     */
	void leaveOneOutReduction();

	/**
	 * Does grow reduction
	 */
	void growthReduction();

    bool equalQueues(
	
		std::priority_queue<std::pair<size_t, double>, 
		std::vector<std::pair<size_t, double> >, PairCmpr>& q1,

		std::priority_queue<std::pair<size_t, double>, 
		std::vector<std::pair<size_t, double> >, PairCmpr>& q2);

	void printQueue(std::priority_queue<std::pair<size_t, double>, 
		std::vector<std::pair<size_t, double> >, PairCmpr>& nearestKInstances);


};

#endif