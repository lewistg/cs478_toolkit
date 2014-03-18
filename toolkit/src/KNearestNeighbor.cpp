#include <utility>
#include <cassert>
#include <cmath>
#include <math.h>
#include <iostream>
#include <map>
#include "KNearestNeighbor.h"

KNearestNeighbor::KNearestNeighbor():_k(5)
{

}

void KNearestNeighbor::train(Matrix& features, Matrix& labels)
{
	_examples = Matrix(features);
	_examples.copyPart(features, 0, 0, features.rows(), features.cols());

	_exampleLabels = Matrix(labels);
	_exampleLabels.copyPart(labels, 0, 0, labels.rows(), labels.cols());

	// normalize the examples
	for(size_t col = 0; col < _examples.cols(); col++)
	{
		_colMaxes.push_back(_examples.columnMax(col));
		_colMins.push_back(_examples.columnMin(col));
	}

	for(size_t row = 0; row < _examples.rows(); row++)
		normalizeFeatures(_examples.row(row));

	_minLabel = labels.columnMin(0);
	_maxLabel = labels.columnMax(0);
}

/*namespace 
{
	struct PairCmpr
	{
		bool operator() (const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) 
		{
			return lhs.second < rhs.second;
		}
	};
};*/

void KNearestNeighbor::predictWithIgnore(const std::vector<double>& features, std::vector<double>& labels, 
	std::vector<bool>* ignoredExamples)
{
	if(ignoredExamples != NULL)
		assert(ignoredExamples->size() < _examples.rows());

	std::vector<double> normFeatures(features);
	normalizeFeatures(normFeatures);

	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr> 
		nearestKInstances;
	for(size_t i = 0; i < _examples.rows(); i++)
	{
		if(ignoredExamples != NULL && ignoredExamples->at(i))
			continue;

		double exampleDist = dist(_examples.row(i), normFeatures);
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
	
	predictNominal(normFeatures, labels, nearestKInstances);
	//regressionPrediction(normFeatures, labels, nearestKInstances);
}

void KNearestNeighbor::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	predictWithIgnore(features, labels);
}

void KNearestNeighbor::predictNominal(const std::vector<double>& normFeatures, std::vector<double>& labels, 
	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr>&
	nearestKInstances)
{
	// take a vote of the closest neighbors
	std::map<double, size_t> labelCounts;
	while(!nearestKInstances.empty())
	{
		size_t nearNeighborIndex = nearestKInstances.top().first;
		nearestKInstances.pop();

		double nearNeighborLabel = _exampleLabels.row(nearNeighborIndex)[0];
		double nearNeighborDist = dist(_examples.row(nearNeighborIndex), normFeatures);
		if(nearNeighborDist <= 0.00000001)
		{
			labels[0] = nearNeighborLabel;
			return;
		}

        // weighted voting
		double weight = 1.0 / (nearNeighborDist * nearNeighborDist);
		labelCounts[nearNeighborLabel] += weight * 1;

		// non-weighted voting
		//labelCounts[nearNeighborLabel] += 1;
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

void KNearestNeighbor::regressionPrediction(const std::vector<double>& normFeatures, std::vector<double>& labels, 
	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr>&
	nearestKInstances)
{
	double labelSum = 0.0;
	double weightSum = 0.0;
	while(!nearestKInstances.empty())
	{
		size_t nearNeighborIndex = nearestKInstances.top().first;
		nearestKInstances.pop();

		double nearNeighborLabel = _exampleLabels.row(nearNeighborIndex)[0];

		// test
		//nearNeighborLabel = (nearNeighborLabel - _minLabel) / (_maxLabel - _minLabel);

		double nearNeighborDist = dist(_examples.row(nearNeighborIndex), normFeatures);
		if(nearNeighborDist <= 0.00000001)
		{
			labels[0] = nearNeighborLabel;
			return;
		}

        // weighted voting
		/*double weight = 1.0 / (nearNeighborDist * nearNeighborDist);
		weightSum += weight;
		labelSum += weight * nearNeighborLabel;*/

		// non-weighted voting
		labelSum += nearNeighborLabel; 
	}

	// non-weighted voting
	double avgLabel = labelSum / static_cast<double>(_k);
	labels[0] = avgLabel;

	/*double avgLabel = labelSum / weightSum; 
	labels[0] = avgLabel;*/
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

void KNearestNeighbor::normalizeFeatures(std::vector<double>& features)
{
	for(size_t col = 0; col < _examples.cols(); col++)
	{
		double delta = _colMaxes[col] - _colMins[col];
		features[col] = (features[col] - _colMins[col]) / delta;
	}
}

void KNearestNeighbor::leaveOneOutReduction()
{
    std::vector<bool> ignoredExample(false, _examples.rows());
	std::vector<double> label(1);
    for(size_t i = 0; i < _examples.rows(); i++)
    {
        ignoredExample[i] = true;
		predictWithIgnore(_examples.row(i), label, &ignoredExample);
        bool correctlyPredicted = (label[0] == _exampleLabels[i][0]);
        if(!correctlyPredicted)
			ignoredExample[i] = false;
    }

	Matrix reducedExamples(_examples);
    Matrix reducedExampleLabels(_exampleLabels);

    for(size_t i = 0; i < _examples.rows(); i++)
    {
        if(!ignoredExample[i])
        {
            reducedExamples.copyRow(_examples.row(i));
            reducedExampleLabels.copyRow(_examples.row(i));
        }
    }

	_examples.clear();
	_examples = Matrix(reducedExamples);
	_examples.copyPart(reducedExamples, 0, 0, reducedExamples.rows(), reducedExamples.cols());

	_exampleLabels.clear();
	_exampleLabels = Matrix(_exampleLabels);
	_exampleLabels.copyPart(reducedExampleLabels, 0, 0, reducedExampleLabels.rows(), reducedExampleLabels.cols());
}
