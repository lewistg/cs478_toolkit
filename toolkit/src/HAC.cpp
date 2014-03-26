#include <iostream>
#include <cassert>
#include <cfloat>
#include <map>
#include <cmath>
#include "HAC.h"
#include "data_utils.h"
#include "ClusteringUtils.h"

#define SINGLE_LINK 0

HAC::HAC(size_t numClusters):_logOn(true), _numClusters(numClusters)
{

}

void HAC::train(Matrix& features, Matrix& labels)
{
	_instanceToCluster = std::vector<size_t>(features.rows()); 
	_adjMatrix = std::vector<std::vector<double> >(features.rows(), std::vector<double>(features.rows(), 0));
	for(size_t i = 0; i < features.rows(); i++)
	{
		_instanceToCluster[i] = i;
		for(size_t j = 0; j < features.rows(); j++)
		{
			double dist = 0;
			if(i != j)
				dist = ClusteringUtils::dist(features, i, j);
			_adjMatrix[i][j] = dist;
		}
	}

	size_t numClusters = features.rows();
	while(numClusters > _numClusters)
	{
		size_t cluster0 = 0;
		size_t cluster1 = 0;

#if SINGLE_LINK
		// find the min-dist (single link)
		double minDist = DBL_MAX;
		for(size_t i = 0; i < features.rows(); i++)
		{
			for(size_t j = i + 1; j < features.rows(); j++)
			{
				if(_adjMatrix[i][j] < minDist && _instanceToCluster[i] != _instanceToCluster[j])
				{
					minDist = _adjMatrix[i][j];
					cluster0 = _instanceToCluster[i];
					cluster1 = _instanceToCluster[j];
				}
			}
		}
		if(_logOn)
		{
			std::cout << "Merging clusters: " << cluster0 << " and " << cluster1 << std::endl;
			std::cout << "Distance: " << minDist << std::endl;
		}
#else // complete link (find the minimum largest distance between clusters)
		std::vector<std::vector<double> > clusterDistances(numClusters, std::vector<double>(numClusters, DBL_MIN));
		for(size_t i = 0; i < features.rows(); i++)
		{
			for(size_t j = i + 1; j < features.rows(); j++)
			{
				if(_instanceToCluster[i] == _instanceToCluster[j])
					continue;

				size_t lowClusterIndex = std::min(_instanceToCluster[i],_instanceToCluster[j]);
				size_t highClusterIndex = std::max(_instanceToCluster[i],_instanceToCluster[j]);

				if(_adjMatrix[i][j] > clusterDistances[lowClusterIndex][highClusterIndex])
					clusterDistances[lowClusterIndex][highClusterIndex] = _adjMatrix[i][j];
			}
		}

        double minLargestDist = DBL_MAX;
		for(size_t i = 0; i < clusterDistances.size(); i++)
		{
			for(size_t j = i + 1; j < clusterDistances.size(); j++)
			{
				if(clusterDistances[i][j] < minLargestDist)
				{
					minLargestDist = clusterDistances[i][j];
					cluster0 = i; 
					cluster1 = j; 
				}
			}
		}

		if(_logOn)
		{
			std::cout << "Merging clusters: " << cluster0 << " and " << cluster1 << std::endl;
			std::cout << "Distance: " << minLargestDist << std::endl;
		}
#endif

		// group clusters
		assert(cluster0 != cluster1);
		for(size_t i = 0; i < _instanceToCluster.size(); i++)
		{
			if(_instanceToCluster[i] == cluster1)
				_instanceToCluster[i] = cluster0;
		}
		std::cout << std::endl;
		
		normalizeLabels();
		numClusters -= 1;
	}


	std::vector<Matrix> clusters(_numClusters, features);
	for(size_t i = 0; i < _instanceToCluster.size(); i++)
	{
		size_t clusterIndex = _instanceToCluster[i];
		clusters[clusterIndex].copyRow(features[i]);
	}

	// calculate centroids
	std::vector<std::vector<double> > clusterMeans(_numClusters, std::vector<double>(features.cols()));
	for(size_t i = 0; i < _numClusters; i++)
	{
		for(size_t j = 0; j < features.cols(); j++)
		{
			if(features.valueCount(j) == 0) // continuous data
				clusterMeans[i][j] = clusters[i].columnMean(j);
			else
				clusterMeans[i][j] = clusters[i].mostCommonValue(j);
		}

		if(_logOn)
		{
			std::cout << "Centroid " << i << ": ";
			std::cout << getInstanceString(clusterMeans[i], features) << std::endl;
		}
	}

	// calculate SSE
	double sse = 0.0;
	for(size_t i = 0; i < _instanceToCluster.size(); i++)
	{
		double d = ClusteringUtils::dist(features, i, clusterMeans[_instanceToCluster[i]]);
		sse += d * d;	
	}

	if(_logOn)
	{
		std::cout << "Total sse: " << sse << std::endl;
	}
}

void HAC::normalizeLabels()
{
	std::map<size_t, size_t> oldClustIdToNew;
	size_t newId = 0;
	for(size_t i = 0; i < _instanceToCluster.size(); i++)
	{
		if(oldClustIdToNew.find(_instanceToCluster[i]) == oldClustIdToNew.end())		
		{
			oldClustIdToNew[_instanceToCluster[i]] = newId;
			newId += 1;
		}

		_instanceToCluster[i] = oldClustIdToNew[_instanceToCluster[i]];
	}
}

void HAC::predict(const std::vector<double>& features, std::vector<double>& labels)
{

}