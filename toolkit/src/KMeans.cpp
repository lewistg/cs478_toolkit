#include <list>
#include <cmath>
#include <cassert>
#include <cfloat>
#include "KMeans.h"
#include "matrix.h"

KMeans::KMeans(size_t numMeans): _numMeans(numMeans)
{

}

KMeans::~KMeans()
{

}

void KMeans::train(Matrix& features, Matrix& labels)
{
	assert(features.rows() > _numMeans);

	std::vector<std::vector<double> > clusterMeans(_numMeans);
	for(size_t i = 0; i < _numMeans; i++)
		clusterMeans[i] = features[i];

	std::vector<size_t> assignedCluster(features.rows(), _numMeans + 10);
	bool converged = false;
	std::vector<Matrix> clusters(_numMeans, features);
	while(!converged)
	{
		converged = true;
		clusters = std::vector<Matrix>(_numMeans, features);

        for(size_t i = 0; i < features.rows(); i++)
		{
			size_t instanceCluster = closestCluster(features, i, clusterMeans);
			if(assignedCluster[i] != instanceCluster)
			{
				clusters[instanceCluster].copyRow(features[i]);
				assignedCluster[i] = instanceCluster;
				converged = false;
			}
		}

		for(size_t i = 0; i < _numMeans; i++)
		{
			for(size_t j = 0; j < features.cols(); j++)
			{
				if(features.valueCount(i) == 0) // continuous data
					clusterMeans[i][j] = clusters[i].columnMean(j);
				else
					clusterMeans[i][j] = clusters[i].mostCommonValue(j);
			}
		}
	}
}

void KMeans::predict(const std::vector<double>& features, std::vector<double>& labels)
{

}

size_t KMeans::closestCluster(Matrix& features, size_t instanceIndex, const std::vector<std::vector<double> >& clusterMeans)
{
	double closestMeanDist = DBL_MAX;
	size_t closestMean = 0;
	for(size_t i = 0; i < clusterMeans.size(); i++)
	{
		double distToMean = dist(features, instanceIndex, clusterMeans[i]);
		if(distToMean < closestMeanDist)
		{
			closestMean = i;	
			closestMeanDist = distToMean;
		}
	}

	return closestMean;
}

double KMeans::dist(Matrix& features, size_t instanceIndex, const std::vector<double>& clusterMean)
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
		else if(features[instanceIndex][i] == clusterMean[i]) // nominal
		{
			deltaSum += 1;
		}
	}

	double dist = sqrt(deltaSum);
	return dist;
}