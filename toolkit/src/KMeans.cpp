#include <list>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <iostream>
#include "KMeans.h"
#include "matrix.h"
#include "data_utils.h"

KMeans::KMeans(size_t numMeans): _numMeans(numMeans),  _log(true)
{

}

KMeans::~KMeans()
{

}

void KMeans::train(Matrix& features, Matrix& labels)
{
	assert(features.rows() > _numMeans);

	_clusterMeans = std::vector<std::vector<double> >(_numMeans);
	for(size_t i = 0; i < _numMeans; i++)
	{
		_clusterMeans[i] = features[i];
		if(_log)
		{
			std::cout << "Centroid " << i << ": ";
			std::cout << getInstanceString(_clusterMeans[i], features) << std::endl;
		}
	}

	std::vector<size_t> assignedCluster(features.rows(), _numMeans + 10);
	_clusters = std::vector<Matrix>(_numMeans, features);

	size_t nItersWithoutImprovement = 0;
	double prevSSE = DBL_MIN;

	while(nItersWithoutImprovement < 3)
	{
		_clusters = std::vector<Matrix>(_numMeans, features);

        for(size_t i = 0; i < features.rows(); i++)
		{
			size_t instanceCluster = closestCluster(features, i, _clusterMeans);
			assignedCluster[i] = instanceCluster;
			_clusters[instanceCluster].copyRow(features[i]);

			if(_log)
				std::cout << i << "=" << instanceCluster << std::endl;
		}

		double sse = calcSSE();
		if(sse > prevSSE)
			nItersWithoutImprovement = 0;
		else
			nItersWithoutImprovement += 1;
		prevSSE = sse;

		if(_log)
			std::cout << "Recomputing cluster means..." << std::endl;

		for(size_t i = 0; i < _numMeans; i++)
		{
			for(size_t j = 0; j < features.cols(); j++)
			{
				if(features.valueCount(j) == 0) // continuous data
					_clusterMeans[i][j] = _clusters[i].columnMean(j);
				else
					_clusterMeans[i][j] = _clusters[i].mostCommonValue(j);
			}

			if(_log)
			{
				std::cout << "Centroid " << i << ": ";
				std::cout << getInstanceString(_clusterMeans[i], features) << std::endl;
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
		else if(features[instanceIndex][i] != clusterMean[i]) // nominal
		{
			deltaSum += 1;
		}
	}

	double dist = sqrt(deltaSum);
	return dist;
}

double KMeans::calcSSE()
{
	double sse = 0.0;
	for(size_t i = 0; i < _numMeans; i++)
	{
		for(size_t j = 0; j < _clusters[i].rows(); j++)
		{
			double sseSqrt = dist(_clusters[i], j, _clusterMeans[i]);
			sse += (sseSqrt * sseSqrt);
		}
	}

	if(_log)
		std::cout << "Sum squared - distance of each row with its centroid = " << sse << std::endl;

	return sse;
}