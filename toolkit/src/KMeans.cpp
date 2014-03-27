#include <list>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <set>
#include "KMeans.h"
#include "matrix.h"
#include "ClusteringUtils.h"
#include "data_utils.h"

KMeans::KMeans(size_t numMeans): _numMeans(numMeans),  _log(true)
{
	assert(_numMeans > 1);
}

KMeans::~KMeans()
{

}

void KMeans::train(Matrix& features, Matrix& labels)
{
	static size_t runNum = 0;
	runNum += 1;

	assert(features.rows() > _numMeans);
	Rand r(time(NULL) + runNum);
	features.shuffleRows(r, &labels);

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

	size_t nIters = 0;
	double prevSSE = DBL_MIN;
	std::set<size_t> clusterNums;

	while(true)
	{
		nIters += 1;

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
		if(sse >= prevSSE && nIters > 3)
		{
			ClusteringUtils::outputClusterStats(_clusters, std::cout);
			break;
		}

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

			/*if(_log)
			{
				std::cout << "Centroid " << i << ": ";
				std::cout << getInstanceString(_clusterMeans[i], features) << std::endl;
			}*/
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
		double distToMean = ClusteringUtils::dist(features, instanceIndex, clusterMeans[i]);
		if(distToMean < closestMeanDist)
		{
			closestMean = i;	
			closestMeanDist = distToMean;
		}
	}

	return closestMean;
}

double KMeans::calcSSE()
{
	double sse = 0.0;
	for(size_t i = 0; i < _numMeans; i++)
	{
		for(size_t j = 0; j < _clusters[i].rows(); j++)
		{
			double sseSqrt = ClusteringUtils::dist(_clusters[i], j, _clusterMeans[i]);
			sse += (sseSqrt * sseSqrt);
		}
	}

	if(_log)
		std::cout << "Sum squared - distance of each row with its centroid = " << sse << std::endl;

	return sse;
}