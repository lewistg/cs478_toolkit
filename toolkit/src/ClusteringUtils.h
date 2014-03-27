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
	 * Calcualtes the distance between two centroids
     */
	double dist(Matrix& features, const std::vector<double>& centroid0, const std::vector<double>& centroid1);

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

    /**
     * Calculates the Davies-Bouldin Index
     */
    double getDaviesBouldin(std::vector<Matrix>& clusters);
};

#endif