#ifndef _CLUSTERINGUTILS_H_
#define _CLUSTERINGUTILS_H_

#include "matrix.h"

namespace ClusteringUtils
{
	/**
	 * Calculates the distance between
     * @param features
     * @param example
     * @return 
     */
	double dist(Matrix& features, size_t instanceIndex, const std::vector<double>& clusterMean);

	/**
	 * Calculates the distance between two instances
	 */
	double dist(Matrix& features, size_t instanceIndex0, size_t instanceIndex1);

	/**
	 * Calculates the centroid value of a cluster
     */
	std::vector<double> getCentroid(Matrix& cluster);

	/**
	 * Calculates the SSE for a cluster
	 */
	double	calcSSE(Matrix& cluster, const std::vector<double>& centroid);

	/**
	 * Outputs the clustering statistics
	 */
	void outputClusterStats(std::vector<Matrix>& clusters, std::ostream& dataOut);
};

#endif