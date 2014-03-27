#include <cassert>
#include <cmath>
#include "data_utils.h"
#include "ClusteringUtils.h"
#include "learner.h"

double ClusteringUtils::dist(Matrix& features, size_t instanceIndex, const std::vector<double>& clusterMean)
{
	return dist(features, features[instanceIndex], clusterMean);
}

double ClusteringUtils::dist(Matrix& features, size_t instanceIndex0, size_t instanceIndex1)
{
	return dist(features, instanceIndex0, features[instanceIndex1]);
}

double ClusteringUtils::dist(Matrix& features, const std::vector<double>& centroid0, const std::vector<double>& centroid1)
{
	assert(features.cols() == centroid0.size() && features.cols() == centroid1.size());
	
	double delta = 0;
	double deltaSum = 0;
	for(size_t i = 0; i < features.cols(); i++)
	{
		if(centroid0[i] == UNKNOWN_VALUE || centroid1[i] == UNKNOWN_VALUE)
		{
			deltaSum += 1;
		}
		else if(features.valueCount(i) == 0) // continuous
		{
			delta = centroid0[i] - centroid1[i];
			deltaSum += (delta * delta);
		}
		else if(centroid0[i] != centroid1[i]) // nominal
		{
			deltaSum += 1;
		}
	}

	double dist = sqrt(deltaSum);
	return dist;
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

	dataOut << "Clusters: " << std::endl;
    double totalSSE = 0.0;
	for(size_t i = 0; i < clusters.size(); i++)
	{
		std::vector<double> centroid = getCentroid(clusters[i]);
        double sse = calcSSE(clusters[i], centroid);

		dataOut<< "(Cluster id = " << i << ", Centroid = [" << getInstanceString(centroid, clusters[i]) << "], ";
		dataOut << "Number of instances = " << clusters[i].rows() << ", " << "Cluster SSE = " << sse << ")";
		dataOut << std::endl << std::endl;

        totalSSE += sse;
	}

	dataOut << "Davies Bouldin Clustering Index: " << getDaviesBouldin(clusters) << std::endl << std::endl;

    dataOut << "Total SSE: " << totalSSE << std::endl;
}

double ClusteringUtils::getDaviesBouldin(std::vector<Matrix>& clusters)
{
	assert(clusters.size() > 0);
	
	std::vector<double> r(clusters.size(), 0);
	double rSum = 0.0;
	for(size_t i = 0; i < clusters.size(); i++)
	{
		std::vector<double> centroidI = getCentroid(clusters[i]);
		double scatterI = calcSSE(clusters[i], centroidI) / clusters[i].rows();
		double maxR = 0.0;
		for(size_t j = 0; j < clusters.size(); j++)
		{
			if(i == j)
				continue;
			
			std::vector<double> centroidJ = getCentroid(clusters[j]);
			double scatterJ = calcSSE(clusters[j], centroidJ) / clusters[j].rows();
			
			double r = (scatterI + scatterJ) / dist(clusters[j], centroidI, centroidJ);
			if(r > maxR)
				maxR = r;
		}
		rSum += maxR;
	}
	
	double dbIndex = rSum / clusters.size();
	return dbIndex;
}