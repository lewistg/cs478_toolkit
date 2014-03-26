#include <cassert>
#include <cmath>
#include "data_utils.h"
#include "ClusteringUtils.h"
#include "learner.h"

double ClusteringUtils::dist(Matrix& features, size_t instanceIndex, const std::vector<double>& clusterMean)
{
	assert(features.cols() == clusterMean.size());
	
	double delta = 0;
	double deltaSum = 0;
	for(size_t i = 0; i < features.cols(); i++)
	{
		if(features[instanceIndex][i] == UNKNOWN_VALUE || clusterMean[i] == UNKNOWN_VALUE)
		{
			deltaSum += 1;
		}
		else if(features.valueCount(i) == 0) // continuous
		{
			delta = features[instanceIndex][i]	- clusterMean[i];
			deltaSum += (delta * delta);
		}
		else if(features[instanceIndex][i] != clusterMean[i]) // nominal
		{
			deltaSum += 1;
		}
	}

	double dist = sqrt(deltaSum);
	return dist;
}

double ClusteringUtils::dist(Matrix& features, size_t instanceIndex0, size_t instanceIndex1)
{
	return dist(features, instanceIndex0, features[instanceIndex1]);
}

std::vector<double> ClusteringUtils::getCentroid(Matrix& cluster)
{
	std::vector<double> centroid(cluster.cols());
	for(size_t j = 0; j < cluster.cols(); j++)
	{
		if(cluster.valueCount(j) == 0) // continuous data
			centroid[j] = cluster.columnMean(j);
		else
			centroid[j] = cluster.mostCommonValue(j);
	}

    return centroid;
}

double ClusteringUtils::calcSSE(Matrix& cluster, const std::vector<double>& centroid)
{
	double sse = 0;
	for(size_t i = 0; i < cluster.rows(); i++)
	{
		double d = dist(cluster, i, centroid);
		sse +=  (d * d);
    }

	return sse;
}

void ClusteringUtils::outputClusterStats(std::vector<Matrix>& clusters, std::ostream& dataOut)
{
	dataOut << "Number of clusters: " << clusters.size() << std::endl;

	dataOut << "Cluster centroids: " << std::endl;
    double totalSSE = 0.0;
	for(size_t i = 0; i < clusters.size(); i++)
	{
		std::vector<double> centroid = getCentroid(clusters[i]);
        double sse = calcSSE(clusters[i], centroid);
		dataOut << "\tCentroid " << i << ": " << getInstanceString(centroid, clusters[i]);
		dataOut << " (number of instances = " << clusters[i].rows() << ", ";
        dataOut << "SSE = " << sse << ")" << std::endl;

        totalSSE += sse;
	}

    dataOut << "Total SSE: " << totalSSE << std::endl;
}