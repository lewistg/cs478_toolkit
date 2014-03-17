#include <utility>
#include <cassert>
#include <cmath>
#include <queue>
#include <math.h>
#include <map>
#include "KNearestNeighbor.h"

KNearestNeighbor::KNearestNeighbor():_k(3)
{

}

void KNearestNeighbor::train(Matrix& features, Matrix& labels)
{
	_examples = Matrix(features);
	_examples.copyPart(features, 0, 0, features.rows(), features.cols());

	_exampleLabels = Matrix(labels);
	_exampleLabels.copyPart(labels, 0, 0, labels.rows(), labels.cols());
}

namespace 
{
	struct PairCmpr
	{
		bool operator() (const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) 
		{
			return lhs.second < rhs.second;
		}
	};
};

void KNearestNeighbor::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr> 
		nearestKInstances;
	for(size_t i = 0; i < _examples.rows(); i++)
	{
		double exampleDist = dist(_examples.row(i), features);
		if(nearestKInstances.size() >= _k)
		{
			if(nearestKInstances.top().second > exampleDist)
			{
				nearestKInstances.push(std::pair<size_t, double>(i, exampleDist));
				if(nearestKInstances.size() > _k)
					nearestKInstances.pop(); 
			}
		}
		else
		{
				nearestKInstances.push(std::pair<size_t, double>(i, exampleDist));
		}
	}

	assert(nearestKInstances.size() == _k);

	// take a vote of the closest neighbors
	std::map<double, size_t> labelCounts;
	while(!nearestKInstances.empty())
	{
		size_t nearNeighborIndex = nearestKInstances.top().first;
		nearestKInstances.pop();

		double nearNeighborLabel = _exampleLabels.row(nearNeighborIndex)[0];
		labelCounts[nearNeighborLabel] += 1;
	}

	size_t maxCount = 0;
	double maxLabel = 0.0;
	for(std::map<double, size_t>::iterator itr = labelCounts.begin(); itr != labelCounts.end(); itr++)
	{
		if(itr->second > maxCount)
		{
			maxCount = itr->second;
			maxLabel = itr->first;
		}
	}
	
	labels[0] = maxLabel;
}

double KNearestNeighbor::dist(const std::vector<double>& features, const std::vector<double>& example)
{
	assert(features.size() == example.size());
	
	double delta = 0;
	double deltaSum = 0;
	for(size_t i = 0; i < features.size(); i++)
	{
		delta = features[i]	- example[i];
		deltaSum += (delta * delta);
	}

	double dist = sqrt(deltaSum);
	return dist;
}