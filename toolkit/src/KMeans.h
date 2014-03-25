#ifndef _KMEANS_H_
#define _KMEANS_H_

#include <vector>
#include "learner.h"

class KMeans: public SupervisedLearner
{
public:
	/**
	 * Constructor
	 */
	KMeans(size_t numMeans);

	/**
	 * Destructor
	 */
	~KMeans();

	/**
	 * @override
	 */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * @override
	 */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

private:
	/**The number of means*/
	size_t _numMeans;
	/**The clusters*/
	std::vector<Matrix> _clusters;
	/**The cluster means*/
	std::vector<std::vector<double> > _clusterMeans;
	/**Turns logging on*/
	bool _log;

	/**
	 * Calculates the closest cluster
     * @param features
     * @param clusterMean
     * @return 
     */
	size_t closestCluster(Matrix& features, size_t instanceIndex, const std::vector<std::vector<double> >& clusterMeans);

	/**
	 * Calculates the SSE for all clusters
	 */
	double calcSSE();
};

#endif