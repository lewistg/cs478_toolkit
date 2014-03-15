#ifndef _KNEARESTNEIGHBOR_H_
#define _KNEARESTNEIGHBOR_H_

#include "learner.h"


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

	/**
	 * Utility function for calculating the distance between a training example
	 * and the given features.
	 */
	double dist(const std::vector<double>& features, const std::vector<double>& example);
};

#endif