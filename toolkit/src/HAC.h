#ifndef _HAC_H_
#define _HAC_H_

#include <vector>
#include "learner.h"


class HAC: public SupervisedLearner
{
public:
	/**
	 * Constructor
     */
	HAC(size_t numClusters);

	/**
	 * @override
	 */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * @override
     */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

private:
	/**Indicates whether or not logging is turned on*/
	bool _logOn;
	/**Desired number of clusters*/
	size_t _numClusters;
	/**Keeps track of the closest two */
	std::vector<std::vector<double> > _adjMatrix; 
	/**Index from cluster to instance*/
	std::vector<std::vector<size_t> > _clusterToInstance;
	/**Instance to cluster*/
	std::vector<size_t> _instanceToCluster;

	/**
	 * Utility function for normalizing labels
	 */
	void normalizeLabels();
};

#endif